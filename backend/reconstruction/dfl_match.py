"""Turn one half of a real DFL match (full-pitch TRACAB tracking, CC BY 4.0) into an app match file.

The file follows the same contract as the simulator / CV output ({meta, players, events, frames}) plus
"alt_frames" {"replay", "live"}. The match itself (frames, analysis) is the ground truth; the camera estimate is kept
for comparison (UI: Truth / Replay / Live and the Error overlay):
  replay / live  what a broadcast camera would show (real SkillCorner camera footprint transferred onto this
                 match), hidden players estimated by the DFL-trained model (fold whose test match this is when
                 possible, so the estimator never saw the match)
  truth          every player's real tracked position

    .venv/bin/python -m backend.reconstruction.dfl_match --match DFL-MAT-J03WMX --half 1
    MATCH_FILE=data/dfl_J03WMX_h1.json ./run.sh
"""
import argparse
import glob
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from backend.reconstruction.datasets import DFL_DIR, L, ROOT, W, dfl_ball_height, dfl_matches, parse_dfl
from backend.reconstruction.estimator import SRC_INF, SRC_OBS
from backend.reconstruction.postprocess import context_blend, estimate_lag, event_pins, jump_stats, limit_speed
from backend.reconstruction.real_benchmark import UNSEEN, formation, mask_sc, split

SECTIONS = {1: "firstHalf", 2: "secondHalf"}
LOMO_FEAT = 2  # feature version of the leave-one-match-out models (v2 beat v1 on 7/7 held-out matches)
REVEAL_MIN_HIDDEN = 5  # s: a reappearance counts as a prediction test if the player was out of view this long
PIN_MODE = "soft"  # event pins: "hard" (anchor), "soft" (pull halfway) or "off"; --tune on validation matches: soft
                   # was marginally best in both folds (replay 7.08->7.07 / 7.16->7.14 m); pins are ~5-6 m off at 1 Hz
CTX_WEIGHT = {"replay": 0.0, "live": 0.0}  # shape-context blend; --tune: any weight > 0 made replay worse in both folds
POSITION = {"TW": "GK", "IVL": "CB", "IVR": "CB", "IVZ": "CB", "LV": "LB", "RV": "RB", "DML": "DM", "DMR": "DM",
            "DMZ": "DM", "DLM": "CM", "DRM": "CM", "ZM": "CM", "ZO": "AM", "OLM": "LM", "ORM": "RM", "LM": "LM",
            "RM": "RM", "LA": "LW", "RA": "RW", "HL": "LW", "HR": "RW", "STZ": "ST", "STL": "ST", "STR": "ST", "HST": "ST"}
COLORS = {"A": "#e11d48", "B": "#38bdf8"}
SHORT = {"1. FC Köln": "KOE", "FC Bayern München": "FCB", "VfL Bochum 1848": "BOC", "Bayer 04 Leverkusen": "B04",
         "Fortuna Düsseldorf": "F95"}


def short(name):
    return SHORT.get(name) or next((w for w in name.split() if len(w) > 3 and w.isalpha()), name)[:3].upper()
SHOT_OUTCOME = {"SuccessfulShot": "goal", "SavedShot": "saved", "ShotWide": "wide", "BlockedShot": "blocked",
                "ShotWoodWork": "post"}


def _ts(s):
    return datetime.fromisoformat(s).timestamp()


def section_start(mid, section):
    """Wall-clock time of the segment's first 1 Hz sample (first frame N of the section, multiple of 25)."""
    f = glob.glob(str(DFL_DIR / f"DFL_04_03_positions_raw_observed_*_{mid}.xml"))[0]
    best = None
    want = False
    with open(f) as fh:
        for line in fh:
            if line.startswith("<FrameSet"):
                want = f'GameSection="{section}"' in line
                continue
            if want:
                m = re.search(r'N="(\d+)" T="([^"]+)"', line)
                if m and (best is None or int(m.group(1)) < best[0]):
                    best = (int(m.group(1)), _ts(m.group(2)))
                want = False
    n0, t0 = best
    n_first = -(-n0 // 25) * 25
    return t0 + (n_first - n0) * 0.04


def raw_events(mid):
    f = glob.glob(str(DFL_DIR / f"DFL_03_02_events_raw_*_{mid}.xml"))[0]
    import xml.etree.ElementTree as ET
    out = []
    for ev in ET.parse(f).getroot().iter("Event"):
        kids = list(ev)
        if not kids:
            continue
        t = _ts(ev.get("CalculatedTimestamp") or ev.get("EventTime"))
        out.append((t, kids[0]))
    return sorted(out, key=lambda e: e[0])


def pick_fold(mid):
    """Model whose held-out test match is `mid` (the estimator never trained on it): the leave-one-match-out fold
    for this match when its weights exist (LOMO_FEAT picks the feature version), else the original 2-fold models."""
    from backend.reconstruction.datasets import dfl_matches
    from backend.reconstruction.inference import WEIGHTS
    from backend.reconstruction.ml import lomo_split
    ms = dfl_matches()
    if mid in ms:  # 7-fold models (6 training matches, refit, v3 features): fold i holds out match i
        name = f"k7r_{ms.index(mid)}_v3"
        if all((WEIGHTS / f"hidden_{m}_{name}.pt").exists() for m in ("live", "replay")):
            return name, True
    for i in range(7):
        if lomo_split(i)[2] == mid:
            name = f"lomo{i}" + ("" if LOMO_FEAT == 1 else f"_v{LOMO_FEAT}")
            if all((WEIGHTS / f"hidden_{m}_{name}.pt").exists() for m in ("live", "replay")):
                return name, True
    if mid == UNSEEN[0]:
        return 0, True
    if mid == UNSEEN[1]:
        return 1, True
    return 0, False


def estimate(info, seg, vis, mode, fold, weights_dir=None):
    """Team-shape prior + DFL-trained correction + calibrated radius (same as the real benchmark)."""
    from backend.reconstruction.inference import load_ml
    from backend.reconstruction.ml import apply_calibration, features, predict
    rec = formation(info, seg, vis, mode)
    on = ~np.isnan(seg["pos"][..., 0])
    try:
        net, cal, fv = load_ml(mode, fold,weights_dir)
    except FileNotFoundError:
        rec.meta["estimator"] = "formation"
        return rec
    X, idx = features(np.where(vis[..., None], seg["pos"], np.nan), vis, on, seg["team"], seg["role"],
                      seg["ball"], seg["poss"], seg["alive"], rec.pos, mode, version=fv)
    if len(idx):
        corr, sig = predict(net, X)
        t, j = idx[:, 0], idx[:, 1]
        corr = np.where((seg["team"][j] == "B")[:, None], -corr, corr)
        rec.pos[t, j] = np.clip(rec.pos[t, j] + corr, [0, 0], [L, W])
        rec.rad[t, j], rec.conf[t, j] = apply_calibration(cal, sig)
    for j, r in enumerate(seg["role"]):
        if r == "GK":
            rows = rec.src[:, j] == SRC_INF
            if seg["team"][j] == "A":
                rec.pos[rows, j, 0] = np.minimum(rec.pos[rows, j, 0], 25)
            else:
                rec.pos[rows, j, 0] = np.maximum(rec.pos[rows, j, 0], L - 25)
    rec.meta["estimator"] = f"ml_{mode}_" + (fold if isinstance(fold, str) else f"f{fold}")
    return rec


def reconstruct_half(info, seg, vis, events, order, col, fold, ball, on, ctx_weight,weights_dir=None):
    """Returns ({mode: rec}, pinned mask, error / jump report vs truth). Truth is only used for the report."""
    from types import SimpleNamespace
    T = len(seg["pos"])
    seg_col = {pid: order[k] for pid, k in col.items()}  # display id -> segment column
    lag = estimate_lag(events, ball, seg_col, vis, seg["pos"])
    pins = event_pins(events, ball, seg_col, on, lag=lag) if PIN_MODE != "off" else {}
    pinned = np.zeros_like(vis)
    obs_pos = seg["pos"].copy()
    for (t, j), xy in pins.items():
        if not vis[t, j]:
            pinned[t, j] = True
            obs_pos[t, j] = xy
    fixed = vis | pinned if PIN_MODE == "hard" else vis
    seg2 = dict(seg, pos=obs_pos)
    roles = seg["role"]
    recs, report = {}, {}
    hid = on & ~vis

    def run_mode(mode, fixed_vis, pins_ok):
        rec = estimate(info, seg2 if PIN_MODE == "hard" else seg, fixed_vis, mode, fold,weights_dir)
        if PIN_MODE == "soft":
            for (t, j), xy in pins.items():
                if pins_ok[t, j]:
                    rec.pos[t, j] = 0.5 * rec.pos[t, j] + 0.5 * np.array(xy)
        rec.pos = limit_speed(rec.pos, fixed_vis, on, mode == "live")
        return rec

    # PREDICTION -> REALITY: what the live (causal) model predicted at the very second a hidden player came back
    # into view. One extra live run with every reappearance second masked: live estimates at t only use <= t, so
    # the value there is exactly the prediction made without seeing the player.
    reveal_list = []
    for j in range(vis.shape[1]):
        hidden = 0
        for t in range(T):
            if not on[t, j]:
                hidden = 0
                continue
            if vis[t, j]:
                if hidden >= REVEAL_MIN_HIDDEN and t >= hidden:
                    reveal_list.append((t, j, hidden))
                hidden = 0
            else:
                hidden += 1
    masked = vis.copy()
    for t, j, _ in reveal_list:
        masked[t, j] = False
    rl = run_mode("live", masked, pinned)
    reveals = []
    for t, j, h in reveal_list:
        pred, act = rl.pos[t, j], seg["pos"][t, j]
        err = float(np.hypot(*(pred - act)))
        reveals.append({"t": int(t), "j": int(j), "hidden_s": int(h), "predicted": [round(float(pred[0]), 1), round(float(pred[1]), 1)],
                        "actual": [round(float(act[0]), 1), round(float(act[1]), 1)], "error_m": round(err, 2),
                        "radius_m": round(float(rl.rad[t, j]), 1), "confidence": round(float(rl.conf[t, j]), 2),
                        "inside_radius": bool(err <= rl.rad[t, j])})
    for mode in ("replay", "live"):
        rec = estimate(info, seg2 if PIN_MODE == "hard" else seg, fixed, mode, fold,weights_dir)
        causal = mode == "live"
        if PIN_MODE == "soft":  # pin pulls the estimate halfway; not an anchor
            for (t, j), xy in pins.items():
                if pinned[t, j]:
                    rec.pos[t, j] = 0.5 * rec.pos[t, j] + 0.5 * np.array(xy)
        steps = {"model": rec.pos.copy()}
        pos = context_blend(rec.pos, fixed, seg["team"], roles, on, seg["poss"], ctx_weight[mode], causal)
        steps["shape_context"] = pos
        pos = limit_speed(pos, fixed, on, causal)
        steps["speed_limited"] = pos
        rec.pos = pos
        if PIN_MODE == "hard":
            rec.conf[pinned] = 0.95
            rec.rad[pinned] = 2.0
        recs[mode] = rec
        rep = {}
        for name, p in steps.items():
            e = np.linalg.norm(p - seg["pos"], axis=2)[hid]
            rep[name] = {"hidden_mae_m": round(float(e.mean()), 2), "within_5m": round(float((e <= 5).mean()), 3),
                         "jumps": jump_stats(p, on, hid)}
        report[mode] = rep
    report["truth_jumps"] = jump_stats(seg["pos"], on, hid)
    if reveals:
        e = np.array([r["error_m"] for r in reveals])
        hs = np.array([r["hidden_s"] for r in reveals])
        report["reveals"] = {"n": len(reveals), "mean_error_m": round(float(e.mean()), 2), "median_error_m": round(float(np.median(e)), 2),
                             "inside_80pct_radius": round(float(np.mean([r["inside_radius"] for r in reveals])), 3),
                             "by_hidden_s": {f"{a}-{b}s": round(float(e[(hs >= a) & (hs < b)].mean()), 2)
                                             for a, b in ((5, 10), (10, 30), (30, 9999)) if ((hs >= a) & (hs < b)).any()}}
    report["_reveals"] = reveals
    report["event_lag_s"] = lag
    report["pin_mode"] = PIN_MODE
    report["pin_error_m"] = round(float(np.mean([np.hypot(*(np.array(xy) - seg["pos"][t, j])) for (t, j), xy in pins.items()])), 2) if pins else None
    return recs, pinned, report


RESTART_NAME = {"KickOff": "Kick-off", "ThrowIn": "Throw-in", "FreeKick": "Free kick", "CornerKick": "Corner",
                "GoalKick": "Goal kick", "Penalty": "Penalty"}


GOAL_HALF_WIDTH = 3.66  # m, half the goal mouth
CROSSBAR = 2.44         # m


def display_ball(ball, alive, events, restarts, T, z=None):
    """Ball for display + per-second dead-ball label (None while the ball is in play).
    Not a goal: the ball shows briefly where play stopped (outside the line if it went out), then is taken to the
    spot where play restarts and waits there until the restart. Goal: it stays in the net until it is carried to
    the centre spot for the kick-off."""
    b = ball.copy()
    dead = [None] * T
    z = np.zeros(T) if z is None else z
    goals = sorted(e["timestamp"] for e in events if e["event_type"] == "GOAL")
    stops = sorted(e["timestamp"] for e in events if e["event_type"] in ("FOUL", "OFFSIDE"))
    t = 0
    while t < T:
        if alive[t]:
            t += 1
            continue
        s = t
        while t < T and not alive[t]:
            t += 1
        e = t  # dead spell [s, e)
        if s == 0 or e - s < 2:
            continue
        stop = b[s - 1].copy()
        over_bar = False
        for k in range(s, min(e, s + 4)):  # crossed a line shortly after the whistle?
            x, y = ball[k]
            if x < -0.3 or x > L + 0.3 or y < -0.3 or y > W + 0.3:
                stop = np.array([np.clip(x, -1.5, L + 1.5), np.clip(y, -1.5, W + 1.5)])
                break
        kind = next((k for tt, k in restarts if s - 1 <= tt <= e + 6), None)  # restart events can be logged late
        if kind is None and any(s - 3 <= tt <= s + 1 for tt in stops):
            kind = "Free kick"
        goal = next((g for g in goals if s - 4 <= g <= s + 2), None)
        behind_goal = (stop[0] < 0 or stop[0] > L) and abs(stop[1] - W / 2) <= GOAL_HALF_WIDTH + 0.3
        if goal is None and behind_goal:
            # crossed the goal line between the posts without a goal: over the bar, or never really out
            # (e.g. the keeper holding it on his line) -> keep it on the pitch
            if z[max(0, s - 3):s + 2].max() > CROSSBAR - 0.2:
                over_bar = True
            else:
                stop[0] = np.clip(stop[0], 0.5, L - 0.5)
        if goal is not None:
            side_x = -1.2 if stop[0] < L / 2 else L + 1.2
            stop = np.array([side_x, np.clip(stop[1], W / 2 - 3.2, W / 2 + 3.2)])
            restart = np.array([L / 2, W / 2])
            move = min(3, max(1, (e - s) // 3))
            b[s:e - move] = stop
            for k in range(move):
                a = (k + 1) / (move + 1)
                b[e - move + k] = stop * (1 - a) + restart * a
            kind = "Goal · kick-off"
        else:
            # where play restarts: the ball's last tracked spot before the restart (where it is placed / thrown from)
            restart = ball[e - 1].copy() if np.isfinite(ball[e - 1]).all() else (ball[e] if e < T else stop)
            if kind == "Throw-in":  # thrown from the touchline
                restart[1] = -0.5 if restart[1] < W / 2 else W + 0.5
            elif kind == "Goal kick":  # placed in the goal area
                restart = np.array([5.5 if restart[0] < L / 2 else L - 5.5, np.clip(restart[1], W / 2 - 9, W / 2 + 9)])
            elif kind == "Corner":
                restart = np.array([0.0 if restart[0] < L / 2 else L, 0.0 if restart[1] < W / 2 else W])
            hold, move = (1, 2) if e - s >= 4 else (0, 1)
            b[s:s + hold] = stop
            for k in range(move):
                a = (k + 1) / move
                b[s + hold + k] = stop * (1 - a) + restart * a
            b[s + hold + move:e] = restart
        for k in range(s, e):
            dead[k] = ("Over the bar · " if over_bar else "") + (kind or "Dead ball")
    return b, dead


def build(mid, half=1, seed=0, ctx_weight=None, fold=None,weights_dir=None):
    ctx_weight = ctx_weight or CTX_WEIGHT
    info, segs = parse_dfl(mid)
    seg = segs[half - 1]
    assert seg["section"] == SECTIONS[half], seg["section"]
    T = len(seg["pos"])
    on = ~np.isnan(seg["pos"][..., 0])
    keep = np.where(on.any(0))[0]  # players who play in this half
    P = info["players"]
    pid_of = {seg["pids"][j]: f"{seg['team'][j]}{P[seg['pids'][j]]['shirt']}" for j in keep}
    team_of = {seg["pids"][j]: seg["team"][j] for j in keep}
    order = sorted(keep, key=lambda j: (seg["team"][j], seg["role"][j] != "GK", not on[0, j], int(P[seg["pids"][j]]["shirt"])))
    order_ids = [pid_of[seg["pids"][j]] for j in order]
    col = {pid: k for k, pid in enumerate(order_ids)}

    players = []
    for j in order:
        p = P[seg["pids"][j]]
        pos = POSITION.get(p["dfl_position"] or "", "CM")
        players.append({"id": pid_of[seg["pids"][j]], "team": seg["team"][j], "shirt": int(p["shirt"]), "name": p["name"],
                        "start_role": pos if on[:5, j].any() else "SUB", "position": pos})

    # camera (real broadcast footprint); reconstruction runs after the event feed is read (event pins)
    vis, quads = mask_sc(seg, seed, return_quads=True)
    vis &= on
    fold, unseen = pick_fold(mid) if fold is None else (fold, mid in UNSEEN)
    if weights_dir is not None:
        from backend.reconstruction.inference import load_ml
        from cv.audit_hidden_sensitivity import require_held_out
        import torch
        for mode in ('live','replay'):
            require_held_out(torch.load(Path(weights_dir)/f'hidden_{mode}_{fold}.pt',weights_only=False),mid)
            load_ml(mode,fold,weights_dir)
        unseen=True
    cut = ~vis.any(1)

    ball = seg["ball"].copy()
    for t in range(1, T):  # ball is lost by tracking for a few samples: hold
        if not np.isfinite(ball[t]).all():
            ball[t] = ball[t - 1]
    ball = np.nan_to_num(ball, nan=0.0)
    ball = np.clip(ball, [-2, -2], [L + 2, W + 2])
    poss = seg["poss"]
    team_arr = np.array([seg["team"][j] for j in order])

    def nearest(t, team, xy=None):
        xy = ball[t] if xy is None else xy
        best, bd = None, 1e9
        for k, j in enumerate(order):
            if team is not None and team_arr[k] != team or not on[t, j]:
                continue
            d = np.hypot(*(seg["pos"][t, j] - xy))
            if d < bd:
                best, bd = order_ids[k], d
        return best, bd

    carrier = []
    for t in range(T):
        tm = {0: "A", 1: "B"}.get(int(poss[t]))
        c, d = nearest(t, tm) if tm and seg["alive"][t] else (None, 1e9)
        carrier.append(c if d < 2.5 else None)

    def frames(rec, bxy=None):
        bxy = ball if bxy is None else bxy
        out = []
        for t in range(T):
            xy, ob, cf, rd = [], [], [], []
            for j in order:
                if rec is None:
                    p = seg["pos"][t, j]
                    ok = on[t, j]
                    xy.append([round(float(p[0]), 1), round(float(p[1]), 1)] if ok else None)
                    ob.append(True if ok else None)
                    continue
                s = rec.src[t, j]
                if s == 0 or not on[t, j]:
                    xy.append(None), ob.append(None), cf.append(None), rd.append(None)
                    continue
                # Serialize the original measurement for an observed player.
                # A double orientation transform can change a .05 rounding tie.
                p = seg["pos"][t, j] if s == SRC_OBS else rec.pos[t, j]
                xy.append([round(float(p[0]), 1), round(float(p[1]), 1)])
                ob.append(bool(s == SRC_OBS))
                cf.append(1.0 if s == SRC_OBS else round(float(rec.conf[t, j]), 2))
                rd.append(0.0 if s == SRC_OBS else round(float(rec.rad[t, j]), 1))
            f = {"t": t, "b": [round(float(bxy[t, 0]), 1), round(float(bxy[t, 1]), 1)],
                 "p": {0: "A", 1: "B"}.get(int(poss[t])), "c": carrier[t], "xy": xy, "obs": ob}
            if rec is not None:
                f.update({"conf": cf, "rad": rd, "cut": bool(cut[t])})
                if np.isfinite(quads[t]).all():
                    f["cam"] = [[round(float(x), 1), round(float(y), 1)] for x, y in quads[t]]  # camera footprint
            if dead[t]:
                f["ds"] = dead[t]  # dead ball: what restarts play (play has not restarted yet)
            out.append(f)
        return out

    # ---- events (official DFL event feed; locations from the tracked ball at that second)
    t0 = section_start(mid, SECTIONS[half])
    events = []
    restarts = []  # (t, name) from the event feed, for the dead-ball label

    def emit(t, et, team, pid, **kw):
        t = float(np.clip(round(t, 1), 0, T - 1))
        ti, a = int(t), t - int(t)
        bx, by = ball[ti] * (1 - a) + ball[min(T - 1, ti + 1)] * a  # ball where the event happened
        e = {"id": "", "timestamp": t, "event_type": et, "team_id": team, "player_id": pid,
             "x": round(float(bx), 1), "y": round(float(by), 1)}
        e.update(kw)
        events.append(e)
        return e

    def who(obj):
        return pid_of.get(obj) if obj else None

    def team_id(club):
        return "A" if club == info["home"] else "B" if club == info["away"] else None

    score = {"A": 0, "B": 0}
    initial = None
    goals_before = {"A": 0, "B": 0}
    for wall, el in raw_events(mid):
        t = wall - t0
        tag = el.tag
        if t < -1 or t > T:
            if t < -1 and half == 2 and tag == "ShotAtGoal" and el.find("SuccessfulShot") is not None:
                goals_before[team_id(el.get("Team"))] += 1
            continue
        play = el if tag == "Play" else el.find("Play")
        if tag in RESTART_NAME:
            restarts.append((t, RESTART_NAME[tag]))
        if tag in ("KickOff", "ThrowIn", "FreeKick", "CornerKick", "GoalKick") and play is not None:
            team = team_id(el.get("Team") or play.get("Team"))
            label = {"KickOff": "KICK_OFF", "ThrowIn": "THROW_IN", "FreeKick": "FREE_KICK", "CornerKick": "CORNER", "GoalKick": None}[tag]
            if label:
                emit(t, label, team, who(play.get("Player")), source="dfl_event")
        if play is not None and (play.find("Pass") is not None or play.find("Cross") is not None):
            pid, rid = who(play.get("Player")), who(play.get("Recipient"))
            team = team_id(play.get("Team"))
            ti = int(np.clip(t, 0, T - 1))
            tgt = ball[min(T - 1, ti + 2)]  # refined below to where the next on-ball action happened
            succ = play.get("Evaluation") in ("successfullyCompleted", "successful")
            sx = ball[ti, 0] if team == "A" else L - ball[ti, 0]
            tx = tgt[0] if team == "A" else L - tgt[0]
            emit(t, "PASS", team, pid, target_x=round(float(tgt[0]), 1), target_y=round(float(tgt[1]), 1),
                 receiver_id=rid if succ else None, success=succ, pressure=0.0,
                 distance=round(float(np.hypot(*(tgt - ball[ti]))), 1), progression=round(float(tx - sx), 1),
                 cross=play.find("Cross") is not None)
        shot = el if tag == "ShotAtGoal" else el.find("ShotAtGoal")
        if shot is not None:
            team = team_id(shot.get("Team"))
            kid = next((k.tag for k in shot if k.tag in SHOT_OUTCOME), None)
            out = SHOT_OUTCOME.get(kid, "off_target")
            xg = round(float(shot.get("xG") or 0), 3)
            pid = who(shot.get("Player"))
            emit(t, "SHOT", team, pid, xg=xg, outcome=out, on_target=out in ("goal", "saved"),
                 distance=round(float(shot.get("DistanceToGoal") or 0), 1))
            if out == "goal":
                score[team] += 1
                emit(t + 0.05, "GOAL", team, pid, assist_id=who(shot.get("AssistShotAtGoal")), xg=xg, score=dict(score))
        if tag == "TacklingGame" and el.get("Winner") in pid_of:
            emit(t, "TACKLE", team_id(el.get("WinnerTeam")), who(el.get("Winner")), opponent_id=who(el.get("Loser")),
                 success=el.get("PossessionChange") == "true")
        if tag == "Foul":
            emit(t, "FOUL", team_id(el.get("TeamFouler")), who(el.get("Fouler")), source="dfl_event")
        if tag == "Offside":
            emit(t, "OFFSIDE", team_id(el.get("Team")), who(el.get("Player")), source="dfl_event")
        if tag == "Caution":
            color = el.get("CardColor", "yellow")
            if color in ("red", "yellowRed") and who(el.get("Player")):  # sent off: the team plays with one fewer
                emit(t, "RED_CARD", team_id(el.get("Team")), who(el.get("Player")),
                     reason="second yellow card" if color == "yellowRed" else el.get("Reason", "red card"))
            else:
                emit(t, "CARD", team_id(el.get("Team")), who(el.get("Player")), label=f"{color.title()} card")
        if tag == "SendingOff" and who(el.get("Player")):
            emit(t, "RED_CARD", team_id(el.get("Team")), who(el.get("Player")), reason=el.get("Reason", "sending off"))
        if tag == "Substitution" and who(el.get("PlayerIn")) in col and who(el.get("PlayerOut")) in col:
            emit(max(t, 0), "SUBSTITUTION", team_id(el.get("Team")), who(el.get("PlayerIn")),
                 player_off_id=who(el.get("PlayerOut")), player_on_id=who(el.get("PlayerIn")),
                 role=POSITION.get(el.get("PlayingPosition") or "", None))
    if half == 2:
        initial = goals_before
        for e in events:
            if e["event_type"] == "GOAL":
                e["score"] = {k: e["score"][k] + goals_before[k] for k in "AB"}
    # possession changes from the tracking possession flag (who controls the ball)
    tackles = [e for e in events if e["event_type"] == "TACKLE" and e["success"]]
    passes = [e for e in events if e["event_type"] == "PASS"]
    prev, last_alive = None, True
    for t in range(T):
        cur = {0: "A", 1: "B"}.get(int(poss[t]))
        if cur and cur != prev:
            if prev is None:
                cause = "kickoff" if t < 3 else "restart"
            elif not last_alive or not seg["alive"][t]:
                cause = "restart"
            elif any(abs(e["timestamp"] - t) <= 2 and e["team_id"] == cur for e in tackles):
                cause = "tackle"
            elif any(0 <= t - e["timestamp"] <= 4 and e["team_id"] == prev and not e["success"] for e in passes):
                cause = "interception"
            else:
                cause = "loose_ball"
            pid, _ = nearest(t, cur)
            emit(t, "POSSESSION_CHANGE", cur, pid, from_team=prev, cause=cause)
            prev = cur
        last_alive = bool(seg["alive"][t])
    # pressure on the carrier (opponent within 3 m), at most every 3 s per presser
    last = {}
    for t in range(T):
        c = carrier[t]
        if not c:
            continue
        cj = order[col[c]]
        for k, j in enumerate(order):
            if team_arr[k] == c[0] or not on[t, j]:
                continue
            d = float(np.hypot(*(seg["pos"][t, j] - seg["pos"][t, cj])))
            if d < 3 and t - last.get(order_ids[k], -99) >= 3:
                last[order_ids[k]] = t
                emit(t, "PRESSURE", team_arr[k], order_ids[k], target_player_id=c, pressure=round(1 - d / 3, 2))
    for e in passes:
        e["pressure"] = round(max([1 - float(np.hypot(*(seg["pos"][int(e["timestamp"]), order[k]] - ball[int(e["timestamp"])]))) / 5
                                   for k in range(len(order)) if team_arr[k] != e["team_id"] and on[int(e["timestamp"]), order[k]]] + [0]), 2)
    pr = {"KICK_OFF": 0, "POSSESSION_CHANGE": 1}
    events.sort(key=lambda e: (e["timestamp"], pr.get(e["event_type"], 2)))
    for i, e in enumerate(events):
        e["id"] = f"e{i:05d}"
    acts = [e for e in events if e["event_type"] in ("PASS", "SHOT", "TACKLE")]
    for a, b in zip(acts, acts[1:]):
        if a["event_type"] == "PASS" and 0 < b["timestamp"] - a["timestamp"] <= 6:
            a["target_x"], a["target_y"] = b["x"], b["y"]
            sx, tx = (a["x"], b["x"]) if a["team_id"] == "A" else (L - a["x"], L - b["x"])
            a["distance"] = round(float(np.hypot(b["x"] - a["x"], b["y"] - a["y"])), 1)
            a["progression"] = round(float(tx - sx), 1)

    # ---- hidden players: camera sightings + event pins -> formation prior + DFL-trained correction
    #      -> usual spot in the current shape / phase -> speed limit (no teleporting)
    recs, pinned, post = reconstruct_half(info, seg, vis, events, order, col, fold, ball, on, ctx_weight,weights_dir)
    zb = dfl_ball_height(mid).get(SECTIONS[half], np.zeros(T))
    zb = np.pad(zb[:T], (0, max(0, T - len(zb))))
    ball_disp, dead = display_ball(ball, seg["alive"], events, restarts, T, zb)

    home, away = info["title"].split(":")
    final = dict(initial or {"A": 0, "B": 0})
    for e in events:
        if e["event_type"] == "GOAL":
            final = e["score"]
    replay = frames(recs["replay"], ball_disp)
    truth = frames(None, ball_disp)
    obs_share = float(vis.sum() / max(1, on.sum()))
    meta = {"seed": None, "flags": [], "duration": T - 1, "pitch": {"length": L, "width": W},
            "teams": {"A": {"name": home.strip(), "short": short(home.strip()), "color": COLORS["A"]},
                      "B": {"name": away.strip(), "short": short(away.strip()), "color": COLORS["B"]}},
            "player_order": order_ids, "final_score": final, "initial_score": initial or {"A": 0, "B": 0},
            "clock_offset": 0.0 if half == 1 else 2700.0, "view": "truth", "vod_available": True,
            "source": {"type": "tracking", "dataset": "DFL / IDSSE open data (Bassek et al., Sci Data 2025), CC BY 4.0",
                       "match_id": mid, "competition": info["competition"], "half": half,
                       "camera": "SkillCorner open-data broadcast footprints (MIT) moved onto this match's ball",
                       "observed_share": round(obs_share, 3), "estimator": recs["replay"].meta["estimator"],
                       "estimator_saw_this_match": not unseen},
            "reconstruction": {"mode": "replay", "retrospective": True, "observed_share": round(obs_share, 3)}}
    reveals = post.pop("_reveals", [])
    for r in reveals:
        r["player_id"] = pid_of[seg["pids"][r.pop("j")]]
    meta["source"]["error_vs_truth"] = post
    meta["source"]["event_pins"] = int(pinned.sum())
    # ground truth is the match (analysis runs on it); the camera estimate is kept to compare against
    return {"meta": meta, "players": players, "events": events, "frames": truth, "reveals": reveals,
            "alt_frames": {"replay": replay, "live": frames(recs["live"], ball_disp)}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default=UNSEEN[0], help=f"one of {dfl_matches()}")
    ap.add_argument("--half", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tune", action="store_true", help="choose the shape-context weight on each fold's validation match")
    a = ap.parse_args()
    if a.tune:
        default_pins = PIN_MODE
        for fold in (0, 1):
            val = split(fold)[1]
            for pm in ("off", "soft", "hard"):
                PIN_MODE = pm  # module global (this block runs at module scope)
                r = [build(val, h, a.seed, {"replay": 0.0, "live": 0.0}, fold)["meta"]["source"]["error_vs_truth"] for h in (1, 2)]
                print(f"fold {fold} val {val} pins={pm}: lag {[x['event_lag_s'] for x in r]} pin_err {[x['pin_error_m'] for x in r]} " + "  ".join(
                    f"{m} {np.mean([x[m]['model']['hidden_mae_m'] for x in r]):.2f}"
                    f"->{np.mean([x[m]['speed_limited']['hidden_mae_m'] for x in r]):.2f}" for m in ("replay", "live")), flush=True)
            PIN_MODE = default_pins
            for w in (0.0, 0.25, 0.5, 0.75):
                r = [build(val, h, a.seed, {"replay": w, "live": w}, fold)["meta"]["source"]["error_vs_truth"] for h in (1, 2)]
                print(f"fold {fold} val {val} w={w}: " + "  ".join(
                    f"{m} {np.mean([x[m]['shape_context']['hidden_mae_m'] for x in r]):.2f}"
                    f"->{np.mean([x[m]['speed_limited']['hidden_mae_m'] for x in r]):.2f}" for m in ("replay", "live")), flush=True)
        raise SystemExit
    out = build(a.match, a.half, a.seed)
    path = ROOT / "data" / f"dfl_{a.match.split('-')[-1]}_h{a.half}.json"
    json.dump(out, open(path, "w"), separators=(",", ":"))
    s = out["meta"]["source"]
    print(f"{path}  {out['meta']['teams']['A']['name']} vs {out['meta']['teams']['B']['name']}  half {a.half}  "
          f"{out['meta']['duration']}s  observed {s['observed_share']:.0%}  events {len(out['events'])}  "
          f"score {out['meta']['final_score']}  error {s['error_vs_truth']}  unseen={not s['estimator_saw_this_match']}")
