"""Steps 6-8: detections.jsonl -> the engine's raw match contract {meta, players, events, frames}.

Teams: k-means on shirt colour per track. Direction: the team defending the left goal is "A"
(attacks +x, like the synthetic engine). Tracks are stitched into player slots, positions are
averaged to 1 Hz, ball ownership gives PASS / POSSESSION_CHANGE / PRESSURE, and SoccerNet
Labels-v2 supplies shots, goals and set pieces.
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from cv.detect_track import OUT
from cv.roles import RoleEstimator, lineup, occupant, smooth_gaps
from cv.soccernet import L, W, game_dir, labels
from cv.event_identity import labelled_scorer
from cv.touch_cues import apply_touch_cues
from cv.temporal_samples import sample_seconds
from cv.track_namespace import qualify_rows

ROOT = Path(__file__).resolve().parents[1]
OWN_DIST = 2.5        # m: nearest player within this distance owns the ball
MAX_SPEED = 9.0       # m/s for track stitching
MIN_SLOT_SECONDS = 8  # drop player slots seen for less than this


def kmeans(X, k, iters=50, seed=0):
    rng = np.random.default_rng(seed)
    C = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d = np.min([((X - c) ** 2).sum(1) for c in C], axis=0)
        C.append(X[rng.choice(len(X), p=d / d.sum())])
    C = np.array(C, dtype=float)
    for _ in range(iters):
        lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), axis=1)
        newC = np.array([X[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    return lab, C


def lab_to_hex(lab):
    bgr = cv2.cvtColor(np.uint8([[lab]]), cv2.COLOR_LAB2BGR)[0, 0]
    return "#{:02x}{:02x}{:02x}".format(int(bgr[2]), int(bgr[1]), int(bgr[0]))


def display_color(lab):
    """Kit colour averaged over sun/shadow is muddy; make it broadcast-graphics bright."""
    bgr = cv2.cvtColor(np.uint8([[lab]]), cv2.COLOR_LAB2BGR)
    h, s_, v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0].astype(int)
    if s_ < 50:  # neutral kit: white-ish or black-ish
        return "#e5e7eb" if v > 140 else "#64748b"  # dark kit: slate, readable on pitch and dark UI
    rgb = cv2.cvtColor(np.uint8([[[h, max(s_, 190), max(v, 220)]]]), cv2.COLOR_HSV2RGB)[0, 0]
    return "#{:02x}{:02x}{:02x}".format(*[int(c) for c in rgb])


def own(team, x, y):
    return (x, y) if team == "A" else (L - x, W - y)


def xg(team, x, y):
    ox, oy = own(team, x, y)
    dx, dy = L - ox, oy - W / 2
    d = math.hypot(dx, dy)
    ang = abs(math.atan2(dy + 3.66, dx) - math.atan2(dy - 3.66, dx))
    return round(1 / (1 + math.exp(-(-0.2 - 0.14 * d + 1.2 * ang))), 3), round(d, 1)


def ball_track(balls):
    """Gate ball candidates by physical reachability; re-lock only on strong, repeated evidence.

    balls: [(t, x, y, conf)] possibly several per frame.
    Returns ({second: [(x, y, conf)]}, [(t, x, y, conf)] accepted at detection rate).
    """
    by_t = defaultdict(list)
    for t, x, y, c in balls:
        by_t[round(t, 2)].append((x, y, c))
    out, acc = defaultdict(list), []
    last, last_t, misses = None, -99.0, 0
    for t in sorted(by_t):
        cands = by_t[t]
        if last is None or t - last_t > 1.5:
            best = max(cands, key=lambda c: c[2])
            if best[2] >= 0.2:
                last, last_t, misses = best, t, 0
                out[int(t)].append(best)
                acc.append((t, *best))
            continue
        gate = 3 + 30 * (t - last_t)  # ~30 m/s max ball speed
        ok = [c for c in cands if math.hypot(c[0] - last[0], c[1] - last[1]) <= gate]
        if ok:
            pick = min(ok, key=lambda c: math.hypot(c[0] - last[0], c[1] - last[1]) - 5 * c[2])
            last, last_t, misses = pick, t, 0
            out[int(t)].append(pick)
            acc.append((t, *pick))
        else:
            misses += 1
            strong = max(cands, key=lambda c: c[2])
            if misses >= 5 and strong[2] >= 0.3:  # we were locked onto a false positive
                last, last_t, misses = strong, t, 0
                out[int(t)].append(strong)
                acc.append((t, *strong))
    return out, acc


def team_of_side_fn(home_away):
    return {home_away[t]: t for t in "AB"}


def rebuild_ball(D, touches, full, bsec, ball_raw, all_labels, start, side_team):
    """1 Hz ball positions consistent with play (see build())."""
    out = [tuple(b) for b in ball_raw]
    src = ["raw"] * (D + 1)
    # 1) touches: carried by the toucher until the next touch; in flight between different players
    for (t1, p1, _, x1, y1, _), (t2, p2, _, x2, y2, _) in zip(touches, touches[1:]):
        if t2 - t1 > 8:
            continue
        for sec in range(int(math.ceil(t1)), int(math.floor(t2)) + 1):
            if not 0 <= sec <= D:
                continue
            flight = float(np.clip(math.hypot(x2 - x1, y2 - y1) / 15.0, 0.3, 3.0))  # passes travel ~15 m/s
            release = t2 - flight if p1 != p2 else t2
            if sec <= release and p1 and (pp := full.get(p1, {}).get(sec)) is not None:
                out[sec] = (float(pp[0]), float(pp[1]))       # on the ball: at his feet
                src[sec] = "carried"
            else:
                f = float(np.clip((sec - release) / max(1e-6, t2 - release), 0, 1))
                r0 = out[int(math.floor(release))] if p1 and int(math.floor(release)) >= 0 and src[int(math.floor(release))] == "carried" else (x1, y1)
                out[sec] = (r0[0] + f * (x2 - r0[0]), r0[1] + f * (y2 - r0[1]))  # pass in flight
                src[sec] = "in_flight"
    # raw stretches: a ball near a player is at his feet; otherwise it stays flagged as uncertain
    for sec in range(D + 1):
        if src[sec] != "raw":
            continue
        cand = [(math.hypot(p[0] - out[sec][0], p[1] - out[sec][1]), p) for ser in full.values() if (p := ser.get(sec)) is not None]
        if cand and min(cand, key=lambda c: c[0])[0] <= 4:
            p = min(cand, key=lambda c: c[0])[1]
            out[sec] = (float(p[0]), float(p[1]))
            src[sec] = "near_player"
    # 2) dead ball: out of play / foul -> restart: the ball stops where play stopped
    win = [a for a in all_labels if start - 5 <= a["t"] <= start + D + 5]
    stops = [a for a in win if a["label"] in ("Ball out of play", "Foul", "Offside")]
    restarts = [a for a in win if a["label"] in ("Throw-in", "Corner", "Direct free-kick", "Indirect free-kick",
                                                  "Kick-off", "Penalty", "Goal kick")]
    for a in stops:
        s0 = int(a["t"] - start)
        nxt = [r for r in restarts if r["t"] > a["t"]]
        if not nxt or not 0 <= s0 <= D:
            continue
        s1 = min(D, int(nxt[0]["t"] - start))
        hold = out[s0]
        for sec in range(s0, s1):
            out[sec] = hold
            src[sec] = "dead_ball"
    # 3) goals: into the net, stay there, centre spot at the kick-off
    for g in [a for a in win if a["label"] == "Goal" and a["team"] in side_team]:
        team = side_team[g["team"]]
        g0 = int(g["t"] - start)
        if not 0 <= g0 <= D:
            continue
        ko = [r for r in win if r["label"] == "Kick-off" and r["t"] > g["t"]]
        g1 = min(D, int(ko[0]["t"] - start)) if ko else D
        shot = out[max(0, g0 - 1)]
        net = (L + 1.0 if team == "A" else -1.0, float(np.clip(shot[1], W / 2 - 3, W / 2 + 3)))
        for sec in range(g0, g1):
            out[sec] = net
            src[sec] = "goal"
        if ko and g1 <= D:
            # restart from the centre spot; hold it there until the first touch after the kick-off
            first = next((tc[0] for tc in touches if tc[0] >= g1), g1 + 5)
            for sec in range(g1, min(D, int(first), g1 + 5) + 1):
                out[sec] = (L / 2, W / 2)
                src[sec] = "kick_off"
    return [(float(np.clip(x, -2, L + 2)), float(np.clip(y, -1, W + 1))) for x, y in out], src


def smooth_ball(pts, win=2):
    """Median filter over contiguous 5 Hz segments: kills single-frame jitter between candidates."""
    P = sorted(pts)
    out = []
    for i, (t, x, y, c) in enumerate(P):
        seg = [q for q in P[max(0, i - win):i + win + 1] if abs(q[0] - t) <= 0.21 * win + 0.01]
        out.append((t, float(np.median([q[1] for q in seg])), float(np.median([q[2] for q in seg])), c))
    return out


def detect_touches(ball_pts, frame_people, track_slot, role_of_slot, pid_of_role, full, D,
                   seen_dist=2.0, est_dist=3.0, min_gap=0.6, min_move=3.0):
    """Ball touches from the 5 Hz ball path: direction/speed kinks or a slow ball under control.

    Credit: nearest *detected* player at that instant (identity via track -> slot -> lineup role),
    else the nearest estimated off-screen player, else nobody (team unknown).
    Returns [(t, pid, team, x, y, confidence)] with consecutive touches by the same player merged.
    """
    P = smooth_ball(ball_pts)
    out = []
    for i in range(1, len(P) - 1):
        (t0, x0, y0, _), (t1, x1, y1, _), (t2, x2, y2, _) = P[i - 1], P[i], P[i + 1]
        if t1 - t0 > 0.61 or t2 - t1 > 0.61:
            continue
        v1 = np.array([x1 - x0, y1 - y0]) / (t1 - t0)
        v2 = np.array([x2 - x1, y2 - y1]) / (t2 - t1)
        s1, s2 = np.linalg.norm(v1), np.linalg.norm(v2)
        ang = math.degrees(math.acos(np.clip(v1 @ v2 / (s1 * s2 + 1e-9), -1, 1))) if s1 > 3 and s2 > 3 else 0
        kink = ang > 45 or abs(s2 - s1) > 8 or (s1 < 2.5 and s2 > 10)
        controlled = s1 < 2.5 and s2 < 2.5
        if not (kink or controlled):
            continue
        sec = int(min(D, max(0, t1)))
        near = [(math.hypot(px - x1, py - y1), tid, dteam) for tid, px, py, dteam, _ in frame_people.get(round(t1, 2), [])]
        pid = team = None
        conf = None
        if near and min(near)[0] <= seen_dist:
            _, tid, team = min(near)  # team from *this* detection's shirt colour (robust to track ID swaps)
            tteam, slot = track_slot.get(tid, (None, None))
            k = role_of_slot.get((team, slot, sec)) if tteam == team else None
            if k is not None:
                pid = pid_of_role(team, k, sec)
            else:  # identity unknown: nearest player of the *right team* by estimated position
                cand = [(math.hypot(p[0] - x1, p[1] - y1), q) for q, ser in full.items() if q[0] == team and (p := ser.get(sec)) is not None]
                pid = min(cand)[1] if cand and min(cand)[0] <= 8 else None
            conf = "seen"
        else:
            cand = [(math.hypot(p[0] - x1, p[1] - y1), q) for q, ser in full.items() if (p := ser.get(sec)) is not None]
            if cand and min(cand)[0] <= est_dist:
                pid = min(cand)[1]
                team, conf = pid[0], "estimated"
        if team is None:
            continue
        if out and out[-1][1] == pid and pid is not None and t1 - out[-1][0] < 0.8:
            continue  # same player still on the ball
        if out and out[-1][1] != pid and (t1 - out[-1][0] < min_gap or math.hypot(x1 - out[-1][3], y1 - out[-1][4]) < min_move):
            continue  # a new player needs time and distance: the ball can't jump between feet
        out.append((round(t1, 2), pid, team, float(x1), float(y1), conf))
    return out


LSCALE = 0.35  # lightness weight in colour features: sun/shadow changes L far more than shirt colour


def color_feat(d):
    """6-D: (most chromatic colour, second colour) in Lab with down-weighted lightness."""
    cs = d.get("colors")
    if cs:
        c1, c2 = cs[0], cs[1] if len(cs) > 1 else cs[0]
        return [c1[0] * LSCALE, c1[1], c1[2], c2[0] * LSCALE, c2[1], c2[2]]
    c = d.get("color")
    return [c[0] * LSCALE, c[1], c[2]] * 2 if c else None


def unscale(c):
    return np.array([c[0] / LSCALE, c[1], c[2]])


def build(run_dir: Path, a_team: str, output_path=None, recon_fold=None, recon_weights=None, sampling='legacy_mean',association='legacy',track_namespace='shot'):
    # argparse supplies numeric folds as strings; inference otherwise interprets
    # "0" as a literal checkpoint tag and silently falls back to smoothing.
    if recon_fold is not None and str(recon_fold).isdecimal():
        recon_fold=f'f{int(recon_fold)}'
    run = json.load(open(run_dir / "run.json"))
    game, half, start, end = run["game"], run["half"], run["start"], run["end"]
    caption = json.loads((game_dir(game) / "Labels-caption.json").read_text())
    rows = [json.loads(l) for l in open(run_dir / "detections.jsonl")]
    namespace_basis=run.get('track_id_namespace','legacy reset-local IDs')
    if track_namespace=='shot' and namespace_basis!='shot-qualified-v1':
        rows=qualify_rows(rows)
        namespace_basis='shot-qualified-v1'
    D = int(end - start)

    # ---- tracks (players + keepers; referees are already removed by the fine-tuned detector)
    tracks = defaultdict(list)
    track_roles = defaultdict(Counter)
    balls = []
    dets_all = []  # (t, tid, x, y, feat, role)
    for r in rows:
        for d in r["dets"]:
            if d["cls"] == "ball":
                balls.append((r["t"] - start, d["px"], d["py"], d["conf"]))
            elif d["cls"] == "person" and d["id"] >= 0:
                f = color_feat(d)
                tracks[d["id"]].append((r["t"] - start, d["px"], d["py"], f))
                track_roles[d["id"]][d.get("role", "player")] += 1
                dets_all.append((round(r["t"] - start, 2), d["id"], d["px"], d["py"], f, d.get("role", "player")))
    tracks = {k: v for k, v in tracks.items() if len(v) >= 3}
    is_gk = {k: track_roles[k]["goalkeeper"] > 0.5 * sum(track_roles[k].values()) for k in tracks}
    if len(tracks) < 6:
        raise SystemExit("Too few player tracks — check the window has live main-camera play.")

    # ---- teams. Two shirt-colour clusters and track-majority voting.
    # Generic person models also include referees/keepers: their contamination
    # and unverified side assignment must be evaluated for the current run.
    X = [(i, d) for i, d in enumerate(dets_all) if d[4] is not None and d[1] in tracks and not is_gk[d[1]]]
    feats = np.array([d[4] for _, d in X])
    lab, C = kmeans(feats, 2)
    mean_x = {j: float(np.mean([d[2] for (_, d), l in zip(X, lab) if l == j])) for j in (0, 1)}
    left_cl = min((0, 1), key=lambda j: mean_x[j])
    cl_team = {left_cl: "A", 1 - left_cl: "B"}
    det_team = {}  # (t, tid) -> team from this detection's own colour
    votes = defaultdict(Counter)
    for (_, d), l in zip(X, lab):
        det_team[(d[0], d[1])] = cl_team[l]
        votes[d[1]][cl_team[l]] += 1
    # colour outliers (shirt far from both team colours): keeper if near a goal, else referee/official.
    # The detector's own goalkeeper/referee classes are weak on older broadcasts, so colour decides.
    tfeat = {k: np.median([f for *_, f in tracks[k] if f is not None], axis=0) for k in tracks if any(f is not None for *_, f in tracks[k])}
    dist = {k: float(np.min(np.linalg.norm(C - f, axis=1))) for k, f in tfeat.items()}
    dv = np.array(list(dist.values()))
    cut = float(np.median(dv) + 3.5 * np.median(np.abs(dv - np.median(dv))))
    track_team = {}
    n_ref = 0
    for k in tracks:
        mx = float(np.mean([p[1] for p in tracks[k]]))
        outlier = dist.get(k, 0) > cut
        if is_gk[k] or (outlier and (mx < 22 or mx > L - 22)):
            is_gk[k] = True
            track_team[k] = "A" if mx < L / 2 else "B"  # keeper: by side
        elif outlier:
            n_ref += 1  # referee / linesman: not a player
        elif votes[k]:
            track_team[k] = votes[k].most_common(1)[0][0]
    print(f"colour outliers: {sum(is_gk.values())} keeper tracks, {n_ref} official tracks (cut {cut:.1f})")
    frame_people = defaultdict(list)  # t -> [(track_id, x, y, team_of_this_detection, is_keeper)]
    for t, tid, x, y, f, role in dets_all:
        if tid in track_team:
            frame_people[t].append((tid, x, y, det_team.get((t, tid), track_team[tid]), is_gk[tid]))
    C = np.vstack([C, C.mean(0)])
    other_cl = 2
    home_away = {"A": a_team, "B": "away" if a_team == "home" else "home"}
    lab_meta, all_labels = labels(game, half)
    names = {"home": lab_meta["gameHomeTeam"], "away": lab_meta["gameAwayTeam"]}
    team_name = {t: names[home_away[t]] for t in "AB"}
    print("team colours:", {t: display_color(unscale(C[j])) for j, t in cl_team.items()}, "referee cluster:", display_color(unscale(C[other_cl])))
    print("A (attacks ->) =", team_name["A"], "| B =", team_name["B"], "  (flip with --a-team)")

    # ---- stitch tracks into player slots
    slots = {"A": [], "B": []}
    for k in sorted(track_team, key=lambda k: tracks[k][0][0]):
        team = track_team[k]
        t0, x0, y0, _ = tracks[k][0]
        best, bd = None, 1e9
        for s in slots[team]:
            lt, lx, ly = s["last"]
            gap = t0 - lt
            if 0 < gap <= 20:  # re-link across short camera cuts
                d = math.hypot(x0 - lx, y0 - ly)
                if d <= min(MAX_SPEED * gap + 4, 22) and d < bd:
                    best, bd = s, d
        if best is None:
            best = {"tracks": []}
            slots[team].append(best)
        best["tracks"].append(k)
        t1, x1, y1, _ = tracks[k][-1]
        best["last"] = (t1, x1, y1)

    # ---- 1 Hz positions per stitched slot (anonymous; identities come from the role estimator)
    slot_series, slot_gk, slot_source_t = {}, {}, {}
    for team in "AB":
        for i, sl in enumerate(slots[team]):
            samples = []
            for k in sl["tracks"]:
                for t, x, y, _ in tracks[k]:
                    if 0 <= t < D + 1:
                        samples.append((t,x,y))
            per,source_times=sample_seconds(samples,D,sampling)
            if len(per) >= 2:
                slot_series[(team, i)] = per
                slot_source_t[(team,i)] = source_times
                slot_gk[(team, i)] = sum(is_gk[k] * len(tracks[k]) for k in sl["tracks"]) > 0.5 * sum(len(tracks[k]) for k in sl["tracks"])

    # ---- ball: best detection per second, short gaps interpolated
    bsec = {}
    bt, ball_pts = ball_track(balls)
    for s_, pts in bt.items():
        if 0 <= s_ <= D:
            bsec[s_] = (float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts])), max(p[2] for p in pts))
    if sampling=='nearest':
        positions,times=sample_seconds([(t,x,y) for t,x,y,_ in ball_pts],D,sampling)
        confidence_at={t:c for t,_,_,c in ball_pts}
        bsec={sec:(float(p[0]),float(p[1]),confidence_at[times[sec]]) for sec,p in positions.items()}
    ball, known = [], []
    last = (L / 2, W / 2)
    ks = sorted(bsec)
    for s in range(D + 1):
        if s in bsec:
            last = bsec[s][:2]
            ball.append(last)
            known.append(True)
            continue
        prev = max([k for k in ks if k < s], default=None)
        nxt = min([k for k in ks if k > s], default=None)
        if prev is not None and nxt is not None and nxt - prev <= 3:
            f = (s - prev) / (nxt - prev)
            ball.append((bsec[prev][0] + f * (bsec[nxt][0] - bsec[prev][0]), bsec[prev][1] + f * (bsec[nxt][1] - bsec[prev][1])))
            known.append(True)
        else:
            ball.append(last)
            known.append(False)

    # ---- provisional possession from anonymous slots (input to the role estimator)
    poss_team = []
    cur = None
    for s in range(D + 1):
        if known[s]:
            c = [(math.hypot(ser[s][0] - ball[s][0], ser[s][1] - ball[s][1]), team) for (team, _), ser in slot_series.items() if s in ser]
            if c and min(c)[0] <= OWN_DIST:
                cur = min(c)[1]
        poss_team.append(cur)

    # ---- role-based identities: lineup roles, observed when matched, model-estimated otherwise
    clock0 = (45 * 60 if half == 2 else 0) + start
    players, order, full, observed = [], [], {}, {}
    roles_by_team, role_of_slot, team_state = {}, {}, {}
    for team in "AB":
        roles, tactic = lineup(game, home_away[team])
        est = RoleEstimator(team, roles,association)
        n_lines = len(tactic.split("-")) - 1
        n = len(roles)
        fwd = np.zeros((D + 1, n, 2))
        seen = np.zeros((D + 1, n), dtype=bool)
        for s in range(D + 1):  # forward pass (causal model)
            dets = [(i, ser[s], "GK" if slot_gk[(tm, i)] else "OUT") for (tm, i), ser in slot_series.items() if tm == team and s in ser]
            opp_dets = [ser[s] for (tm, _), ser in slot_series.items() if tm != team and s in ser]
            pv = 0 if poss_team[s] is None else (1 if poss_team[s] == team else -1)
            fwd[s], seen[s] = est.step(s, np.array(ball[s]), pv, dets, opp_dets)
            for k in range(n):
                if seen[s, k]:
                    role_of_slot[(team, est.slot_of[k], s)] = k
        roles_by_team[team] = roles
        team_state[team] = (roles, fwd, seen, n_lines)

    def coarse_role(r, n_lines):
        if r["role"] == "GK":
            return "GK"
        lat = abs(r["anchor"][1]) / 22
        if r["line"] == 1:
            return "FB" if lat > 0.6 else "CB"
        if r["line"] == n_lines:
            return "ST" if lat < 0.5 else "W"
        if r["line"] == n_lines - 1 and n_lines >= 4:
            return "W" if lat > 0.6 else "AM"
        return "WM" if lat > 0.6 else ("DM" if r["line"] == 2 and n_lines >= 4 else "CM")

    def assemble(pins):
        """Hidden players: learned replay model (trained on DFL real tracking, cv/weights/hidden_replay_f0.pt)
        when available, else offline smoothing. Only matched detections are observations;
        touches are soft estimated cues and never overwrite measured player feet."""
        from backend.reconstruction.inference import ml_available, reconstruct_ml
        full_, obs_, conf_, rad_, plist, order_ = {}, {}, {}, {}, [], []
        teams_rows = [(team, k) for team in "AB" for k in range(len(team_state[team][0]))]
        n = len(teams_rows)
        xy = np.full((D + 1, n, 2), np.nan)
        vis_all = np.zeros((D + 1, n), bool)
        for c, (team, k) in enumerate(teams_rows):
            roles, fwd, seen, _ = team_state[team]
            xy[seen[:, k], c] = fwd[seen[:, k], k]
            vis_all[:, c] = seen[:, k]
        cue_rows={(sec,teams_rows.index((tm,k))):p for (tm,k,sec),p in pins.items()}
        use_ml = ml_available(recon_fold,recon_weights)
        if use_ml:
            from types import SimpleNamespace
            order_c = [f"{tm}:{k}" for tm, k in teams_rows]
            poss_code = np.array([{"A": 0, "B": 1}.get(pt, -1) for pt in poss_team])
            mm = SimpleNamespace(xy=xy, ball=np.array(ball, dtype=float), poss=poss_code, order=order_c, T=D + 1,
                                 idx={o: c for c, o in enumerate(order_c)}, players=[], events=[])
            role_d = {o: ("GK" if team_state[tm][0][k]["role"] == "GK" else "CM") for o, (tm, k) in zip(order_c, teams_rows)}
            anchors = {o: tuple(team_state[tm][0][k]["anchor"]) for o, (tm, k) in zip(order_c, teams_rows)}
            rc = np.array([coarse_role(team_state[tm][0][k], team_state[tm][3]) for tm, k in teams_rows])
            tarr = np.array([tm for tm, _ in teams_rows])
            from backend.reconstruction.inference import VIDEO_RADIUS_SCALE
            rec = reconstruct_ml(mm, vis_all, "replay", np.ones_like(vis_all), role_d, anchors, tarr, rc,
                                 radius_scale=VIDEO_RADIUS_SCALE,fold=recon_fold,weights_dir=recon_weights)
            from backend.reconstruction.postprocess import limit_speed  # estimates never outrun a sprint
            est, cf, rd = limit_speed(apply_touch_cues(rec.pos,vis_all,cue_rows), vis_all, np.ones_like(vis_all), causal=False), rec.conf, rec.rad
        for c, (team, k) in enumerate(teams_rows):
            roles, fwd, seen, n_lines = team_state[team]
            if use_ml:
                sm = est[:, c]
            else:
                sm = smooth_gaps(np.where(vis_all[:, c:c + 1], xy[:, c], fwd[:, k]), vis_all[:, c])
                cues={(sec,0):p for (sec,column),p in cue_rows.items() if column==c}
                sm=apply_touch_cues(sm[:,None,:],vis_all[:,c:c+1],cues)[:,0]
            r = roles[k]
            for sec in range(D + 1):
                pl = occupant(r, clock0 + sec)
                pid = f"{team}{pl['shirt_number']}"
                if pid not in full_:
                    full_[pid], obs_[pid], conf_[pid], rad_[pid] = {}, {}, {}, {}
                    pos_label = "GK" if r["role"] == "GK" else ("DEF" if r["line"] == 1 else "FWD" if r["line"] == n_lines else "MID")
                    plist.append({"id": pid, "team": team, "shirt": int(pl["shirt_number"]), "name": pl["short_name"],
                                  "start_role": pos_label, "position": pos_label, "role_slot": r["role"]})
                    order_.append(pid)
                full_[pid][sec] = sm[sec]
                if seen[sec, k]:
                    obs_[pid][sec] = sm[sec]
                elif use_ml:
                    conf_[pid][sec], rad_[pid][sec] = float(cf[sec, c]), float(rd[sec, c])
        assemble.conf, assemble.rad, assemble.used_ml = conf_, rad_, use_ml
        return full_, obs_, plist, order_

    full, observed, players, order = assemble({})
    n_obs = sum(len(v) for v in observed.values())
    n_all = sum(len(v) for v in full.values())

    # ---- touches at detection rate (5 Hz): kinks in the ball path next to a player
    events = []

    def emit(t, et, team, pid, x=None, y=None, **kw):
        e = {"id": "", "timestamp": round(float(t), 2), "event_type": et, "team_id": team, "player_id": pid}
        if x is not None:
            e["x"], e["y"] = round(float(x), 1), round(float(y), 1)
        e.update(kw)
        events.append(e)
        return e

    track_slot = {k: (team, i) for team in "AB" for i, sl in enumerate(slots[team]) for k in sl["tracks"]}

    def pid_of_role(team, k, sec):
        return f"{team}{occupant(roles_by_team[team][k], clock0 + sec)['shirt_number']}"

    touches = detect_touches(ball_pts, frame_people, track_slot, role_of_slot, pid_of_role, full, D)

    # a player who touches the ball is at the ball: pin touchers, then re-smooth everyone
    def role_index(pid, sec):
        team = pid[0]
        for k, r in enumerate(roles_by_team[team]):
            if f"{team}{occupant(r, clock0 + sec)['shirt_number']}" == pid:
                return k
        return None

    pins = {}
    for tt, pid, team, x, y, conf in touches:
        sec = int(min(D, max(0, round(tt))))
        k = role_index(pid, sec) if pid else None
        if k is not None:
            pins[(team, k, sec)] = (x, y)
    full, observed, players, order = assemble(pins)
    from backend.reconstruction.replay_bridge import bridge
    bridge_input=np.array([[full[pid].get(s,[np.nan,np.nan]) for pid in order] for s in range(D+1)])
    bridge_visible=np.array([[s in observed[pid] for pid in order] for s in range(D+1)])
    bridge_positions,bridge_report=bridge(bridge_input,bridge_visible)
    for s in range(D+1):
        for j,pid in enumerate(order):
            if not bridge_visible[s,j] and np.isfinite(bridge_positions[s,j]).all():full[pid][s]=bridge_positions[s,j]

    # ball path from football logic: at the feet of whoever has it, or travelling between touches;
    # in the net after a goal until the labelled kick-off; dead (stationary) between out-of-play and restart
    ball, ball_src = rebuild_ball(D, touches, full, bsec, ball, all_labels, start, team_of_side_fn(home_away))
    # touches -> passes / interceptions
    owner = None  # (pid, team, t, x, y, conf)
    for tt, pid, team, x, y, conf in touches:
        if owner and pid != owner[0] and tt - owner[2] <= 6:
            a, b = owner[1], team
            dist = math.hypot(x - owner[3], y - owner[4])
            if dist >= 3:
                prog = own(a, x, y)[0] - own(a, owner[3], owner[4])[0]
                emit(owner[2], "PASS", a, owner[0], owner[3], owner[4], target_x=round(x, 1), target_y=round(y, 1),
                     receiver_id=pid if a == b else None, success=a == b, pressure=0.0, distance=round(dist, 1),
                     progression=round(float(prog), 1), touch_confidence=min(conf, owner[5], key=lambda c: c == "seen"))
                if a != b:
                    emit(tt, "POSSESSION_CHANGE", b, pid, x, y, from_team=a, cause="interception")
        elif owner is None:
            emit(tt, "POSSESSION_CHANGE", team, pid, x, y, from_team=None, cause="kickoff")
        owner = (pid, team, tt, x, y, conf)

    frames, press_log = [], {}
    ti = 0
    cur = None
    for s in range(D + 1):
        while ti < len(touches) and touches[ti][0] < s + 1:
            cur = touches[ti]
            ti += 1
        carrier = cur[1] if cur and s - cur[0] <= 1.5 else None
        poss = cur[2] if cur else None
        if carrier and carrier in full and (op := full[carrier].get(s)) is not None:
            for pid in order:
                if pid[0] != carrier[0] and (p := observed[pid].get(s)) is not None:
                    d = math.hypot(p[0] - op[0], p[1] - op[1])
                    if d < 4 and s - press_log.get(pid, -99) >= 8:
                        press_log[pid] = s
                        emit(s, "PRESSURE", pid[0], pid, *p, target_player_id=carrier, pressure=round(1 - d / 4, 2))
        xy = [[round(float(p[0]), 1), round(float(p[1]), 1)] if (p := full[pid].get(s)) is not None else None for pid in order]
        ob = [(s in observed[pid]) if s in full[pid] else None for pid in order]
        cf = [None if o is None else (1.0 if o else round(assemble.conf.get(pid, {}).get(s, 0.5), 2)) for pid, o in zip(order, ob)]
        rd = [None if o is None else (0.0 if o else round(assemble.rad.get(pid, {}).get(s, 15.0), 1)) for pid, o in zip(order, ob)]
        frames.append({"t": s, "b": [round(float(ball[s][0]), 1), round(float(ball[s][1]), 1)], "p": poss, "c": carrier, "xy": xy, "obs": ob, "bs": ball_src[s],
                       **({"conf": cf, "rad": rd} if assemble.used_ml else {})})

    # ---- SoccerNet labels -> events
    team_of_side = {home_away[t]: t for t in "AB"}
    score0 = {"A": 0, "B": 0}
    prior = [a for a in all_labels if a["t"] < start]
    if half == 2:
        prior += labels(game, 1)[1]
    for a in prior:
        if a["label"] == "Goal" and a["team"] in team_of_side:
            score0[team_of_side[a["team"]]] += 1
    score = dict(score0)
    win = [a for a in all_labels if start <= a["t"] <= end]
    goals = [a for a in win if a["label"] == "Goal"]
    used_goals = set()
    simple = {"Foul": "FOUL", "Corner": "CORNER", "Throw-in": "THROW_IN", "Offside": "OFFSIDE",
              "Direct free-kick": "FREE_KICK", "Indirect free-kick": "FREE_KICK", "Clearance": "CLEARANCE",
              "Ball out of play": "BALL_OUT", "Yellow card": "CARD", "Red card": "CARD", "Yellow->red card": "CARD",
              "Penalty": "PENALTY", "Goal kick": "GOAL_KICK", "Kick-off": "KICK_OFF", "Substitution": "SUBSTITUTION_LABEL"}

    def shooter(team, s):
        recent = [tc for tc in touches if tc[2] == team and tc[1] and s - 2.5 <= tc[0] <= s + 0.5]
        if recent:
            return recent[-1][1]
        s = min(D, max(0, s))
        bx, by = ball[s]
        c = [(math.hypot(p[0] - bx, p[1] - by) + (0 if s in observed[pid] else 5), pid)
             for pid in order if pid[0] == team and (p := full[pid].get(s)) is not None]
        return min(c)[1] if c else None

    def goal_events(team, s, bx, by, loc_ok=True, goal_second=None):
        nonlocal score
        goal_second = s if goal_second is None else goal_second
        factual = labelled_scorer(caption, home_away[team], half, start+goal_second)
        pid = f"{team}{factual['shirt_number']}" if factual else None
        scorer = {"scorer_verified": factual is not None,
                  "scorer_name": factual.get("long_name",factual["name"]) if factual else None,
                  "identity_source": "Labels-caption.json lineup goal facts" if factual else "unknown"}
        x_g, dist = xg(team, bx, by)
        score = {**score, team: score[team] + 1}
        emit(s, "SHOT", team, pid, bx, by, xg=x_g, outcome="goal", on_target=True, distance=dist,
             location_known=loc_ok, source="soccernet_label", **scorer)
        emit(goal_second, "GOAL", team, pid, bx, by, assist_id=None, xg=x_g, score=dict(score), source="soccernet_label", **scorer)

    def ball_near(s):
        """Nearest actually-detected ball within +-3 s (shots often happen during camera cuts)."""
        for d in (0, -1, 1, -2, 2, -3, 3):
            k = s + d
            if 0 <= k <= D and k in bsec:
                return bsec[k][0], bsec[k][1], True
        b = ball[min(D, max(0, s))]
        return b[0], b[1], False

    for a in win:
        team = team_of_side.get(a["team"])
        s = int(a["t"] - start)
        bx, by, known_loc = ball_near(s)
        if a["label"] in ("Shots on target", "Shots off target") and team:
            goal = next((i for i, g in enumerate(goals) if g["team"] == a["team"] and 0 <= g["t"] - a["t"] <= 3 and i not in used_goals), None)
            if goal is not None:
                used_goals.add(goal)
                goal_events(team, a["t"]-start, bx, by, known_loc, goal_second=goals[goal]["t"]-start)
            else:
                x_g, dist = xg(team, bx, by)
                outcome = "saved" if a["label"] == "Shots on target" else "off_target"
                emit(s, "SHOT", team, shooter(team, s), bx, by, xg=x_g, outcome=outcome,
                     on_target=outcome == "saved", distance=dist, location_known=known_loc, source="soccernet_label")
        elif a["label"] == "Goal" and team:
            i = goals.index(a)
            if i not in used_goals:
                used_goals.add(i)
                goal_events(team, s, bx, by, known_loc, goal_second=a["t"]-start)
        elif a["label"] in simple:
            emit(s, simple[a["label"]], team, None, bx, by, label=a["label"], source="soccernet_label")
    events.sort(key=lambda e: e["timestamp"])
    for i, e in enumerate(events):
        e["id"] = f"e{i:05d}"

    meta = {
        "seed": None, "flags": [], "duration": D, "pitch": {"length": L, "width": W},
        "teams": {t: {"name": team_name[t], "short": team_name[t][:3].upper(),
                      "color": display_color(unscale(C[[j for j, tt in cl_team.items() if tt == t][0]])),
                      "kit_color": lab_to_hex(unscale(C[[j for j, tt in cl_team.items() if tt == t][0]]))} for t in "AB"},
        "player_order": order, "final_score": score, "initial_score": score0,
        "date": game.split("/")[-1].split(" - ")[0], "score_known": True,
        "clock_offset": (45 * 60 if half == 2 else 0) + start,
        "source": {"type": "computer_vision", "game": game, "half": half, "start": start, "end": end,
                   "model": run["model"], "fps": run["fps"],
                   "sampling":sampling,"observation_timing":"Nearest measured sample within 0.25 s; source timestamps retained" if sampling=='nearest' else "Mean over [second, second+1); mean source time retained",
                   "role_association":association,
                   "track_id_namespace":namespace_basis,
                   "tracker_configuration":run.get('tracker',run.get('tracker_path','cv/botsort_broadcast.yaml')),
                   "measurement_kind":run.get('measurement','tracker-filtered box foot'),
                   "calibration_offset_s":run.get('calibration_offset_s',0.),
                   "calibration_sampling":run.get('calibration_sampling','retrospective nearest 2Hz CCBV'),
                   "camera_run":str(run_dir.resolve()) if run.get('camera_H_coordinates')=='normalised_0_to_1_image_to_pitch_metres' else None,
                   "coverage": round(float(np.mean([any(bool(o) for o in f["obs"]) for f in frames])), 3),
                   "observed_share": round(n_obs / max(1, n_all), 3),
                   "identity": "role-based from official lineup (Labels-caption.json)",
                   "identity_verified":False,"mode":"replay","retrospective":True,"replay_bridge":bridge_report,
                   "reconstruction_fold":recon_fold,"reconstruction_weights":str(recon_weights) if recon_weights else None,
                   "uncertainty_basis":"tracking calibration transferred to video; identity and calibration error remain unverified",
                   "offscreen_model": (f"{recon_weights or 'cv/weights'}/hidden_replay_{recon_fold or 'final_v3'}.pt (tracking-domain model; video radius transferred)" if assemble.used_ml
                                       else "smoothing (no learned model found)")},
    }
    out = {"meta": meta, "players": players, "events": events, "frames": frames}
    source_lookup={}
    for (team,slot,sec),role in role_of_slot.items():
        source_lookup[(sec,pid_of_role(team,role,sec))]=slot_source_t.get((team,slot),{}).get(sec)
    for frame in frames:
        frame['observed_t']=[source_lookup.get((frame['t'],pid)) if frame['obs'][j] else None for j,pid in enumerate(order)]
    path = Path(output_path) if output_path else ROOT / "data" / f"cv_{run_dir.name}.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    json.dump(out, open(path, "w"))
    try:  # camera view per second + prediction -> reality tests (cv/clip_extras.py)
        from cv.clip_extras import add_extras
        add_extras(path)
    except Exception as e:
        print("clip extras skipped:", e)
    print(f"players: A={sum(p['team'] == 'A' for p in players)} B={sum(p['team'] == 'B' for p in players)} "
          f"| live coverage {meta['source']['coverage']:.0%} | observed positions {n_obs / max(1, n_all):.0%} (rest estimated) | events: {len(events)} | score {score0} -> {score}")
    print("wrote", path)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", help="cv/out/<slug> (default: most recent)")
    ap.add_argument("--a-team", choices=["home", "away"], default="home",
                    help="which side defends the LEFT goal in this half (check the printed colours)")
    ap.add_argument("--out", help="isolated candidate match file; preserves active exports")
    ap.add_argument("--recon-fold",help="candidate checkpoint tag, e.g. selected")
    ap.add_argument("--recon-weights",help="isolated candidate checkpoint directory")
    ap.add_argument('--sampling',choices=['legacy_mean','nearest'],default='legacy_mean')
    ap.add_argument('--association',choices=['legacy','stable_track'],default='legacy',help='Stable anonymous track aliases; does not verify shirt identity')
    ap.add_argument('--track-namespace',choices=['shot','legacy'],default='shot',help='Separate reset-local IDs across camera gaps using cached flags; legacy reproduces the former unsafe grouping')
    a = ap.parse_args()
    rd = Path(a.run_dir) if a.run_dir else max(OUT.glob("*/run.json"), key=lambda p: p.stat().st_mtime).parent
    build(rd, a.a_team, a.out, a.recon_fold, a.recon_weights,a.sampling,a.association,a.track_namespace)
