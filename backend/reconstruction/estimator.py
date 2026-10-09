"""RECONSTRUCT: hidden player state from partial observations (team-level, deterministic).

Inputs the estimator may use (all observable in a broadcast):
  visible player positions, ball position, possession, lineup roles/formation at kick-off,
  substitutions / red cards (events), and its own past estimates.
It never reads hidden ground-truth positions.

Per team and second:
  1. fit the team block (centre c, scale s) to visible players via role anchors: obs ~= c + s * anchor
  2. adapt each anchor online (EMA) when the player is seen with enough teammates to fix the block
  3. hidden player = c + s * anchor + (last seen offset from the block, decaying) + (short velocity carry)
     keepers: keeper model near the goal line, shading to the ball
  4. no teammate visible -> block drifts towards a ball/possession prior learned online

LIVE mode uses only the past. REPLAY mode (offline VOD) also runs the same estimator backwards in time
from the next sighting and fuses both by inverse variance, then enforces reachability (8.5 m/s) from the
last and next sightings. Uncertainty radius / confidence come from a calibration table measured on a
*different* synthetic match (see evaluate.py).
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from backend.models.match import L, W, MatchData
from simulator.scenarios import FORMATIONS

CALIB = Path(__file__).resolve().parents[2] / "data" / "eval" / "calibration.json"
EVIDENCE = ["observed", "previous_position", "velocity", "team_shape", "ball_position", "player_role", "future_reappearance"]
EV = {k: 1 << i for i, k in enumerate(EVIDENCE)}
DT_BUCKETS = [1, 2, 4, 7, 12, 20, 40, 1e9]  # upper bounds (s) since last / until next sighting
VMAX = 8.5
OFFSET_TAU = 40.0   # s: how quickly a hidden player's estimate blends from "last seen + block shift" to his role slot
SHIFT_GAIN = 0.8    # share of the team-block movement since last seen applied to a hidden player
SHIFT_ADAPT = True  # scale the gain by how much of the team is visible (a biased ball-side subset -> trust less)
SHIFT_AXES = (1.0, 1.0)  # per-axis multiplier (length, width)
SRC_OFF, SRC_OBS, SRC_INF = 0, 1, 2


N_KEYS = (len(DT_BUCKETS) + 1) * 3


def vis_bucket(n_team_visible):
    return 0 if n_team_visible <= 2 else (1 if n_team_visible <= 5 else 2)


def calib_lookup(table, key):
    """Composite (time-hidden x team-visibility) stats, falling back to the time-only pool when sparse."""
    row = table.get(str(key))
    if row and row["n"] >= 80:
        return row["r80"], row["p5"]
    pool = table.get(f"d{key // 3}")
    if pool and pool["n"] > 0:
        return pool["r80"], pool["p5"]
    return 15.0, 0.2


def bucket(dt):
    for i, b in enumerate(DT_BUCKETS):
        if dt <= b:
            return i
    return len(DT_BUCKETS) - 1


def load_calibration():
    try:
        return json.load(open(CALIB))
    except Exception:
        return None


@dataclass
class Reconstruction:
    mode: str
    pos: np.ndarray        # (T, N, 2) world metres
    src: np.ndarray        # (T, N) 0 off pitch, 1 observed, 2 inferred
    conf: np.ndarray       # (T, N) P(error <= 5 m), 1 for observed
    rad: np.ndarray        # (T, N) uncertainty radius (m, calibrated ~80% coverage), 0 for observed
    evid: np.ndarray       # (T, N) bitmask over EVIDENCE
    key: np.ndarray        # (T, N) calibration key (bucket index), -1 observed/off
    meta: dict = field(default_factory=dict)

    def evidence_list(self, t, j):
        return [k for k in EVIDENCE if self.evid[t, j] & EV[k]]


def _own(team, p):
    p = np.asarray(p, dtype=float)
    if team == "A":
        return p
    q = p.copy()
    q[..., 0] = L - q[..., 0]
    q[..., 1] = W - q[..., 1]
    return q


def _template(role, formation):
    """Lineup prior: own-frame offset of a role from the team centre."""
    ln, lat = FORMATIONS.get(formation, FORMATIONS["4-4-2"]).get(role, (2, 0))
    return np.array([(ln - 1) / 2 * 30 - 15, lat * 22])


def roster_timeline(m: MatchData):
    """On-pitch flags from lineup + observable events (subs / red cards), not from ground truth."""
    T, N = m.T, len(m.order)
    on = np.zeros((T, N), dtype=bool)
    role = {p["id"]: p.get("start_role") or p.get("position") for p in m.players}
    for j, pid in enumerate(m.order):
        if role.get(pid) not in (None, "SUB"):
            on[:, j] = True
    for e in m.events:
        t = int(e["timestamp"])
        if e["event_type"] == "SUBSTITUTION":
            off, onp = m.idx[e["player_off_id"]], m.idx[e["player_on_id"]]
            on[t:, off] = False
            on[t:, onp] = True
            role[e["player_on_id"]] = e.get("role") or role.get(e["player_off_id"])
        elif e["event_type"] == "RED_CARD":
            on[t:, m.idx[e["player_id"]]] = False
    return on, role


class _TeamFilter:
    def __init__(self, team, idx, roles, formation, calib_live, anchors=None):
        self.team, self.idx = team, idx
        self.gk = [r == "GK" for r in roles]
        self.anchor = np.array(anchors if anchors is not None else [_template(r, formation) for r in roles], dtype=float)
        self.c = None
        self.s = np.array([1.0, 1.0])
        self.c_vel = np.zeros(2)
        self.last = {}         # j -> (t, pos_own, offset_from_block, vel_rel)
        self.prior = np.array([[0.5, 4.0, 22.0], [0.5, 0.0, 17.0]])  # c = a*ball + b*poss + k, refined online
        self._xtx = np.eye(3) * 1e-3
        self._xty = np.zeros((3, 2))
        self.calib = calib_live

    def _radius(self, dt, never):
        k = (len(DT_BUCKETS) if never else bucket(dt)) * 3 + vis_bucket(self.n_out)
        if self.calib:
            r80, p5 = calib_lookup(self.calib, k)
            return r80, p5, k
        r = 14.0 if never else min(18.0, 2 + 1.6 * np.sqrt(dt))
        return r, float(np.exp(-r / 8)), k

    def step(self, t, obs_own, on_mask, ball_own, poss):
        """obs_own: {j: pos} visible this second (own frame). Returns {j: (pos, conf, rad, evid, key)}."""
        n_obs = len(obs_own)
        prior_c = self.prior[:, 0] * ball_own + self.prior[:, 1] * poss + self.prior[:, 2]
        if self.c is None:
            self.c = prior_c.copy()
        c_prev = self.c.copy()
        outfield = [j for j in obs_own if not self.gk[self.idx.index(j)]]
        self.n_out = len(outfield)
        if len(outfield) >= 2:
            P = np.array([obs_own[j] for j in outfield])
            A = np.array([self.anchor[self.idx.index(j)] for j in outfield])
            if len(outfield) >= 5:  # enough spread to fit the scale too
                for ax in (0, 1):
                    X = np.stack([np.ones(len(A)), A[:, ax]], 1)
                    coef, *_ = np.linalg.lstsq(X, P[:, ax], rcond=None)
                    self.s[ax] = float(np.clip(0.7 * self.s[ax] + 0.3 * coef[1], 0.6, 1.5))
            c_fit = np.median(P - self.s * A, axis=0)
            w = min(1.0, len(outfield) / 6)  # few players: blend with prediction
            self.c = w * c_fit + (1 - w) * (self.c + self.c_vel)
            if len(outfield) >= 6:  # learn the ball/possession -> block prior from well-observed seconds
                x = np.array([ball_own[0], poss, 1.0])
                xy = np.array([ball_own[1], poss, 1.0])
                self._xtx += np.outer(x, x)
                self._xty[:, 0] += x * self.c[0]
                self._xty[:, 1] += xy * self.c[1]
                if self._xtx[0, 0] > 50:
                    sol = np.linalg.solve(self._xtx + np.eye(3) * 1e-3, self._xty)
                    self.prior[0] = sol[:, 0]
                    self.prior[1] = [sol[0, 1], 0.0, sol[2, 1]]
        else:  # nobody to fit (camera cut / far side): hold the block, damp its motion, lean on the ball prior
            self.c_vel *= 0.5
            self.c = self.c + self.c_vel
            if self._xtx[0, 0] > 50:  # only once the prior has been learned from this match
                self.c += 0.05 * (prior_c - self.c)
        self.c_vel = 0.7 * self.c_vel + 0.3 * (self.c - c_prev)
        # anchors adapt when the block is well determined
        if len(outfield) >= 5:
            for j in outfield:
                k = self.idx.index(j)
                rel = (obs_own[j] - self.c) / self.s
                self.anchor[k] += 0.02 * (rel - self.anchor[k])
        out = {}
        for k, j in enumerate(self.idx):
            if not on_mask[j]:
                continue
            if j in obs_own:
                p = obs_own[j]
                prev = self.last.get(j)
                vel = (p - prev[1]) / max(1, t - prev[0]) if prev and t - prev[0] <= 2 else np.zeros(2)
                off = p - (self.c + self.s * self.anchor[k])
                self.last[j] = (t, p.copy(), off, self.c.copy())
                out[j] = (p, 1.0, 0.0, EV["observed"], -1)
                continue
            ev = EV["ball_position"] | EV["player_role"] | (EV["team_shape"] if n_obs >= 3 else 0)
            prev = self.last.get(j)
            if self.gk[k]:
                gx = float(np.clip(4 + 0.12 * ball_own[0], 2.5, 16))
                gy = float(np.clip(W / 2 + 0.3 * (ball_own[1] - W / 2), W / 2 - 8, W / 2 + 8))
                p = np.array([gx, gy])
                if prev and t - prev[0] <= 3:
                    p = 0.5 * p + 0.5 * prev[1]
                    ev |= EV["previous_position"]
                dt, never = (t - prev[0]) if prev else 60, prev is None
            elif prev is None:
                p = self.c + self.s * self.anchor[k]
                dt, never = 60, True
            else:
                dt, never = t - prev[0], False
                g = SHIFT_GAIN * (min(1.0, self.n_out / 8) if SHIFT_ADAPT else 1.0)
                base = prev[1] + g * np.array(SHIFT_AXES) * (self.c - prev[3])  # where he was, moved with his team
                slot = self.c + self.s * self.anchor[k]                # where his role says he should be
                w = 1 - np.exp(-dt / OFFSET_TAU)
                p = (1 - w) * base + w * slot
                ev |= EV["previous_position"]
                reach = VMAX * dt  # can't be further than he could run since last seen
                d = p - prev[1]
                r = np.linalg.norm(d)
                if r > reach:
                    p = prev[1] + d / r * reach
            p = np.clip(p, [0.5, 0.5], [L - 0.5, W - 0.5])
            rad, conf, key = self._radius(dt, never)
            out[j] = (p, conf, rad, ev, key)
        return out


def _run(m: MatchData, vis, on, role, calib, reverse=False, anchors=None, formations=None):
    T, N = vis.shape
    order = list(range(T))[::-1] if reverse else list(range(T))
    pos = np.full((T, N, 2), np.nan)
    conf = np.zeros((T, N))
    rad = np.zeros((T, N))
    evid = np.zeros((T, N), dtype=np.int16)
    key = np.full((T, N), -1, dtype=np.int16)
    src = np.zeros((T, N), dtype=np.int8)
    poss_code = m.poss
    filters = {}
    for team in ("A", "B"):
        idx = [j for j, pid in enumerate(m.order) if pid.startswith(team)]
        form = (formations or {}).get(team) or ("4-3-3" if team == "A" else "4-4-2")  # kick-off formations (lineup knowledge)
        roles = [role.get(m.order[j]) or "CM" for j in idx]
        anc = [anchors[m.order[j]] for j in idx] if anchors else None
        filters[team] = _TeamFilter(team, idx, roles, form, calib, anc)
    for step_i, t in enumerate(order):
        for team, f in filters.items():
            code = 0 if team == "A" else 1
            ball_own = _own(team, m.ball[t])
            poss = 0.0 if poss_code[t] < 0 else (1.0 if poss_code[t] == code else -1.0)
            obs = {j: _own(team, m.xy[t, j]) for j in f.idx if vis[t, j] and on[t, j]}
            res = f.step(step_i, obs, on[t], ball_own, poss)
            for j, (p, c, r, ev, k) in res.items():
                pos[t, j] = _own(team, p)  # own<->world is an involution
                conf[t, j], rad[t, j], evid[t, j], key[t, j] = c, r, ev, k
                src[t, j] = SRC_OBS if ev & EV["observed"] else SRC_INF
    return pos, conf, rad, evid, key, src


def _dt_arrays(vis, on):
    """Seconds since last / until next sighting for every (t, j)."""
    T, N = vis.shape
    since = np.full((T, N), np.inf)
    until = np.full((T, N), np.inf)
    last = np.full(N, -np.inf)
    for t in range(T):
        last = np.where(vis[t], t, last)
        since[t] = t - last
    nxt = np.full(N, np.inf)
    for t in range(T - 1, -1, -1):
        nxt = np.where(vis[t], t, nxt)
        until[t] = nxt - t
    return since, until


def reconstruct(m: MatchData, vis: np.ndarray, mode: str = "replay", calib=None, on=None, role=None,
                anchors=None, formations=None) -> Reconstruction:
    """mode: 'live' (causal) or 'replay' (retrospective, uses the next sighting too).
    Real data: pass `on` (who is on the pitch), `role` (pid -> role, 'GK' marks keepers) and `anchors`
    (pid -> own-frame offset from the team centre) instead of the simulator's lineup/events."""
    calib = calib if calib is not None else load_calibration()
    if on is None or role is None:
        on, role = roster_timeline(m)
    vis = vis & on
    cl = calib.get("live") if calib else None
    pos, conf, rad, evid, key, src = _run(m, vis, on, role, cl, anchors=anchors, formations=formations)
    if mode == "live":
        return Reconstruction("live", pos, src, conf, rad, evid, key, {"retrospective": False})
    bpos, bconf, brad, bevid, bkey, _ = _run(m, vis, on, role, cl, reverse=True, anchors=anchors, formations=formations)
    since, until = _dt_arrays(vis, on)
    hidden = (src == SRC_INF)
    both = hidden & np.isfinite(until) & np.isfinite(since)
    only_future = hidden & np.isfinite(until) & ~np.isfinite(since)
    wf = 1 / np.maximum(rad, 0.5) ** 2
    wb = 1 / np.maximum(brad, 0.5) ** 2
    fused = (pos * wf[..., None] + bpos * wb[..., None]) / (wf + wb)[..., None]
    pos = np.where(both[..., None], fused, pos)
    pos = np.where(only_future[..., None], bpos, pos)
    # reachability from both sightings
    T = vis.shape[0]
    for t, j in zip(*np.where(both)):
        for anchor_t, dt in ((t - since[t, j], since[t, j]), (t + until[t, j], until[t, j])):
            a = m.xy[int(anchor_t), j]
            d = pos[t, j] - a
            r = np.linalg.norm(d)
            if r > VMAX * dt:
                pos[t, j] = a + d / r * VMAX * dt
    evid = np.where(both | only_future, evid | bevid | EV["future_reappearance"], evid)
    # replay uncertainty: calibrated on the gap position (time to the *nearer* sighting)
    rk = np.where(np.isfinite(np.minimum(since, until)), np.minimum(since, until), 1e9)
    team_of = np.array([pid[0] for pid in m.order])
    nvis = np.zeros(vis.shape, dtype=int)
    for team in ("A", "B"):
        cols = team_of == team
        nvis[:, cols] = vis[:, cols].sum(1, keepdims=True)
    vb = np.where(nvis <= 2, 0, np.where(nvis <= 5, 1, 2))
    keyr = np.where(hidden, np.vectorize(bucket)(rk) * 3 + vb, -1).astype(np.int16)
    cr = calib.get("replay") if calib else None
    if cr:
        lut = np.array([calib_lookup(cr, k) for k in range(N_KEYS)])
        rad = np.where(hidden, lut[np.clip(keyr, 0, None), 0], rad)
        conf = np.where(hidden, lut[np.clip(keyr, 0, None), 1], conf)
    else:
        rad = np.where(hidden, np.minimum(rad, 1 / np.sqrt(wf + wb)), rad)
    return Reconstruction("replay", pos, src, conf, rad, evid, keyr, {"retrospective": True})
