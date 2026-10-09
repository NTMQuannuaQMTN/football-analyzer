"""Lineups -> 11 role slots per team, and the role-based estimator that fills in off-screen players.

Identity is *role-based*: each second, visible detections are assigned to the lineup roles
(Hungarian matching against where each role is expected to be). Real shirt numbers and names
come from SoccerNet `Labels-caption.json`; the rest of the team is estimated with the
Metrica-trained off-screen model (cv/offscreen_model.py).
"""
import json
import re

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from cv.offscreen_model import L, W, centroid_prior, features, load_model
from cv.soccernet import game_dir

LINE_X = {2: [-15, 15], 3: [-16, 0, 16], 4: [-17, -6, 6, 17], 5: [-18, -9, 0, 9, 18]}
LINE_WIDTH = [40, 36, 34, 30, 20]
GK_ANCHOR = (-38.0, 0.0)


def lineup(game: str, side: str):
    """[{role, shirt, name, line, anchor(own frame rel. centroid), on: [(game_sec, ...)]}] + subs."""
    d = json.load(open(game_dir(game) / "Labels-caption.json"))
    lu = d["lineup"][side]
    lines = [int(x) for x in lu["tactic"].split("-")][1:]  # drop the keeper
    starters = sorted([p for p in lu["players"] if p["starting"]], key=lambda p: p["lineup"])
    roles = [{"role": "GK", "line": 0, "anchor": GK_ANCHOR, "players": [(0, starters[0])]}]
    k = 1
    xs = LINE_X[len(lines)]
    for li, n in enumerate(lines):
        w = LINE_WIDTH[min(li, len(LINE_WIDTH) - 1)] if n > 1 else 0
        for j in range(n):
            y = w / 2 - j * (w / (n - 1)) if n > 1 else 0.0  # lineup order is right -> left (+y = right)
            roles.append({"role": f"L{li + 1}{j + 1}", "line": li + 1, "anchor": (xs[li], y), "players": [(0, starters[k])]})
            k += 1
    # substitutions: a bench player's fact {description: "<replaced player>", time: "61'"}
    subs = []
    for p in lu["players"]:
        if p["starting"]:
            continue
        for f in p["facts"]:
            m = re.match(r"^(\d+)(?:\+(\d+))?'$", f.get("time", "").strip())
            if m:
                subs.append(((int(m.group(1)) + int(m.group(2) or 0) - 1) * 60, f["description"], p))
    for t, replaced, p in sorted(subs, key=lambda s: s[0]):
        for r in roles:
            if r["players"][-1][1]["name"] == replaced:
                r["players"].append((t, p))
    return roles, lu["tactic"]


def smooth_gaps(pos, obs, vmax=8.0, lead_tau=15.0):
    """Use *future* sightings too (offline video): nobody teleports to the ball.

    pos: (T, 2) forward estimates (observed rows are measurements), obs: (T,) bool.
    - gap between two sightings: spread the forward model's end error linearly across the gap,
      so the path starts at the last sighting and ends exactly at the next one;
    - every hidden second must be reachable from both sightings at <= vmax m/s;
    - before the first sighting: pull estimates towards it (decaying with time distance).
    """
    pos = np.array(pos, dtype=float)
    T = len(pos)
    idx = np.where(obs)[0]
    if len(idx) == 0:
        return pos
    # leading gap: only a future sighting
    first = idx[0]
    if first > 0:
        err = pos[first] - pos[first - 1]
        for k in range(first):
            pos[k] += err * np.exp(-(first - 1 - k) / lead_tau)
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a <= 1:
            continue
        n = b - a
        err = pos[b] - pos[b - 1]                  # how far the forward guess ended from the reappearance
        for k in range(a + 1, b):
            pos[k] += err * (k - a) / n
            # reachability from both sightings
            for anchor, dt in ((pos[a], k - a), (pos[b], b - k)):
                d = pos[k] - anchor
                r = np.linalg.norm(d)
                if r > vmax * dt:
                    pos[k] = anchor + d / r * vmax * dt
    # final speed limit on estimated seconds (forward then backward), measurements untouched
    for order in (range(1, T), range(T - 2, -1, -1)):
        for k in order:
            if obs[k]:
                continue
            prev = pos[k - 1] if order.step == 1 else pos[k + 1]
            d = pos[k] - prev
            r = np.linalg.norm(d)
            if r > vmax:
                pos[k] = prev + d / r * vmax
    return np.clip(pos, [0, 0], [L, W])


def occupant(role, game_sec):
    cur = role["players"][0][1]
    for t, p in role["players"]:
        if game_sec >= t:
            cur = p
    return cur


def keeper_position(ball_own):
    x = float(np.clip(4 + 0.12 * ball_own[0], 2.5, 16))
    y = float(np.clip(W / 2 + 0.3 * (ball_own[1] - W / 2), W / 2 - 8, W / 2 + 8))
    return np.array([x, y])


def to_own(team, p):
    p = np.asarray(p, dtype=float)
    return p if team == "A" else np.array([L - p[0], W - p[1]])


class RoleEstimator:
    """Per team: track 11 roles through the clip; observed when matched, model-estimated otherwise."""

    GATE = 60.0  # place every visible detection (exact positions); identity may still be approximate
    CONTINUITY = 25.0

    def __init__(self, team: str, roles):
        self.team = team
        self.roles = roles
        self.model, self.coef = load_model()
        self.last = [None] * len(roles)       # (pos_own, t, ball_own)
        self.slot_of = [None] * len(roles)    # stitched track slot last matched to each role
        self.est = None
        self.block = None                     # fitted formation offset (team centroid, own frame)

    def fit_block(self, own_pts, fallback):
        """Shift of the formation template that best explains the visible players (ICP-style)."""
        anchors = np.array([r["anchor"] for r in self.roles])
        if len(own_pts) < 3:
            return fallback
        P = np.array(own_pts)
        off = P.mean(0) - anchors.mean(0)
        for _ in range(6):
            C = np.linalg.norm(P[:, None] - (anchors[None] + off), axis=2)
            r, c = linear_sum_assignment(C)
            new = np.mean(P[r] - anchors[c], axis=0)
            if np.allclose(new, off, atol=0.05):
                break
            off = new
        return off

    def step(self, t, ball_w, poss, dets, opp_dets):
        """dets: [(slot_id, world_xy)] for this team. Returns (world positions (n,2), observed (n,) bool)."""
        b = to_own(self.team, ball_w)
        hints = {sid: d[2] for d in dets if len(d) > 2 and d[2] is not None for sid in [d[0]]}  # e.g. read shirt number
        own = [(d[0], to_own(self.team, d[1])) for d in dets]
        opp = [to_own(self.team, p) for p in opp_dets]
        tm = np.mean([p for _, p in own], axis=0) if own else b
        op = np.mean(opp, axis=0) if opp else b
        anchors = np.array([r["anchor"] for r in self.roles])
        prior_off = centroid_prior(b, poss, self.coef)
        if self.block is None:
            self.block = prior_off
        self.block = self.fit_block([p for _, p in own], self.block + 0.3 * (prior_off - self.block))
        feats, bases = [], []
        for i, r in enumerate(self.roles):
            if self.last[i] is None:
                lp = self.block + anchors[i]
                dt, never, bl = 60, 1, b
            else:
                lp, t0, bl = self.last[i]
                dt, never = t - t0, 0
            feats.append(features(lp, dt, never, b, bl, tm, len(own), op, len(opp), r["anchor"], poss))
            bases.append(lp)
        with torch.no_grad():
            delta = self.model(torch.from_numpy(np.stack(feats))).numpy() * 20
        pred = np.clip(np.stack(bases) + delta, [0, 0], [L, W])
        # long-unseen roles: blend the model's guess with the fitted formation slot
        for i in range(len(self.roles)):
            dt = 60 if self.last[i] is None else t - self.last[i][1]
            if self.roles[i]["role"] == "GK":
                if dt > 2:  # keeper model: near his line, shading to the ball side — never with the outfield block
                    pred[i] = keeper_position(b)
                continue
            w = min(1.0, dt / 20)
            pred[i] = (1 - 0.5 * w) * pred[i] + 0.5 * w * (self.block + anchors[i])
        observed = np.zeros(len(self.roles), dtype=bool)
        if own:
            C = np.zeros((len(own), len(self.roles)))
            for a, (sid, p) in enumerate(own):
                for i, r in enumerate(self.roles):
                    c = np.linalg.norm(p - pred[i])
                    if self.slot_of[i] == sid:
                        c -= self.CONTINUITY
                    h = hints.get(sid)
                    if isinstance(h, (int, np.integer)):
                        c += 0 if h == i else 200  # known identity (jersey number) overrides geometry
                    elif h == "GK":
                        c += 0 if r["role"] == "GK" else 200  # detector says goalkeeper
                    elif h == "OUT" and r["role"] == "GK":
                        c += 200
                    if r["role"] == "GK" and h != "GK" and not (p[0] < 17 and abs(p[1] - W / 2) < 21):
                        c += 200  # a keeper role never takes an outfield detection outside his box
                    if r["role"] == "GK" and p[0] > 22:
                        c += 40
                    if r["role"] != "GK" and p[0] < 6:
                        c += 15
                    C[a, i] = c
            rows, cols = linear_sum_assignment(C)
            for a, i in zip(rows, cols):
                if C[a, i] <= self.GATE:
                    sid, p = own[a]
                    pred[i] = p
                    observed[i] = True
                    self.last[i] = (p, t, b)
                    self.slot_of[i] = sid
        self.est = pred
        world = np.array([to_own(self.team, p) for p in pred])  # rotation is its own inverse
        return world, observed
