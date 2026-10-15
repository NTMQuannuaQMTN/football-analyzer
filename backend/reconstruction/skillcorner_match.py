"""SkillCorner Open Data (A-League 2024/25, MIT) half -> app match file, hidden players reconstructed.

Observed positions are SkillCorner's broadcast detections (is_detected) at 1 Hz, kept exactly. Every undetected
on-pitch player is estimated by our DFL-trained model (formation prior + learned correction + calibrated radius +
speed limit). SkillCorner's own extrapolations are NOT used. The file has:
  frames            replay reconstruction (retrospective: may use the next sighting)  -> analytics run on this
  alt_frames.live   live reconstruction (past only)
  alt_frames.replay same as frames
Events come from SkillCorner's dynamic events (provider computer vision, not an official event feed):
player_possession -> PASS / SHOT, on_ball_engagement -> PRESSURE, regains -> POSSESSION_CHANGE, score changes ->
GOAL (attributed to that team's last shot within 60 s, else marked timing_approximate), substitutions from the
match sheet. xG is this repo's simulator xG model (distance / angle), not a provider value.

    .venv/bin/python -m backend.reconstruction.skillcorner_match --model r13_final --all
"""
import argparse
import csv
import glob
import json
import os
from pathlib import Path

import numpy as np

from backend.reconstruction import hidden_models as hm
from backend.reconstruction.datasets import L, SC_DIR, W
from backend.reconstruction.estimator import SRC_INF
from backend.reconstruction.gsr_benchmark import estimate, model_path
from backend.reconstruction.postprocess import limit_speed
from backend.reconstruction.skillcorner_benchmark import load_match

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
POSITION = {"GK": "GK", "CB": "DEF", "LCB": "DEF", "RCB": "DEF", "LB": "DEF", "RB": "DEF", "LWB": "DEF", "RWB": "DEF",
            "DM": "MID", "LDM": "MID", "RDM": "MID", "LM": "MID", "RM": "MID", "AM": "MID", "LW": "FWD", "RW": "FWD",
            "LF": "FWD", "RF": "FWD", "CF": "FWD", "SUB": "SUB"}
COLORS = ("#f97316", "#38bdf8")


def _xg(own_xy):
    from simulator.match_generator import xg_model
    return float(xg_model(np.asarray(own_xy, float)))


def reconstruct(clip, models, speed=True):
    """{mode: (pos, r80, p5, vis, on)} with GK zone clamp and speed limit, same as the benchmark."""
    vis = np.isfinite(clip["pos"][..., 0]) & clip["on"]
    out = {}
    for mode in ("live", "replay"):
        pos, r80, p5 = estimate(clip, vis, mode, models[mode])
        inf = clip["on"] & ~vis
        for j, r in enumerate(clip["role"]):
            if r == "GK":
                if clip["team"][j] == "A":
                    pos[inf[:, j], j, 0] = np.minimum(pos[inf[:, j], j, 0], 25)
                else:
                    pos[inf[:, j], j, 0] = np.maximum(pos[inf[:, j], j, 0], L - 25)
        if speed:
            pos = limit_speed(pos, vis, clip["on"], mode == "live")
        pos[inf] = np.clip(pos[inf], [0, 0], [L, W])  # smoothing next to an off-pitch measurement: estimates stay on the pitch
        pos = np.where(clip["on"][..., None], pos, np.nan)
        out[mode] = (pos, r80, p5, vis)
    return out


def display_ball(clip, goal_events):
    """Display ball + dead-ball labels, same rules as the DFL exports (dfl_match.display_ball), driven by the provider's
    stoppage labels: the ball is shown where play stopped (outside the line if it went out), then placed at the
    restart spot (touchline / corner flag / goal area / penalty spot / where the free kick is taken) until play
    restarts. After a goal it stays in the net until it is carried to the centre spot for the kick-off. Only the
    displayed ball is changed; the tracked ball is kept in ball_observed / bs."""
    ball, seen = clip["ball"], clip["ball_seen"]
    b = ball.copy()
    T = len(b)
    labels, restart_t, basis = [None] * T, [None] * T, ["annotation_interval"] * T
    outside = lambda p: p[0] < -0.3 or p[0] > L + 0.3 or p[1] < -0.3 or p[1] > W + 0.3
    spells = []
    goals_t = sorted((g["timestamp"], g) for g in goal_events)
    for sp in clip.get("stoppages") or []:
        s0, e0, kind, after = sp
        inside = [g for t, g in goals_t if s0 - 15 <= t < e0]
        if kind == "Goal · kick-off" and not inside:
            kind = "Dead ball"  # provider goal label without a counted goal: never drawn as a goal
        if inside and kind != "Goal · kick-off" and inside[0]["timestamp"] > s0 + 1:
            # set-piece goal: [stop, goal) keeps its restart type, then the goal stoppage until the kick-off
            g = inside[0]
            spells.append((s0, g["timestamp"], kind, after))
            spells.append((g["timestamp"] + 1.0, e0, "Goal · kick-off", "goal inside a set-piece stoppage"))
            continue
        spells.append((s0, e0, "Goal · kick-off" if inside else kind, after))
    spells.sort()
    for i, (s0, e0, k, a) in enumerate(spells):
        nxt = spells[i + 1][0] if i + 1 < len(spells) else T
        if k == "Goal · kick-off" and e0 - s0 < 15:
            # the provider logged a touch right after the goal: run to the kick-off (ball tracked on the centre spot)
            ko = next((t for t in range(int(s0) + 15, min(T, int(s0) + 240)) if seen[t]
                       and np.hypot(ball[t][0] - L / 2, ball[t][1] - W / 2) <= 3), None)
            if ko is not None:
                e0 = float(ko)
        spells[i] = (s0, min(e0, nxt), k, a)  # spells never overlap: the next stoppage ends this one
    for g in goal_events:
        if any(k == "Goal · kick-off" and s0 - 15 <= g["timestamp"] < e0 for s0, e0, k, _ in spells):
            continue
        # no stoppage holds this goal: dead until the ball is next tracked on the centre spot, else until play restarts
        t0 = int(g["timestamp"]) + 15
        ko = next((k for k in range(t0, min(T, t0 + 240)) if seen[k] and np.hypot(ball[k][0] - L / 2, ball[k][1] - W / 2) <= 3), None)
        if ko is None:
            ko = next((st for st in clip.get("possession_starts") or [] if st >= t0), None)
        if ko is not None:
            spells = [x for x in spells if not (g["timestamp"] <= x[0] < ko)]
            spells.append((g["timestamp"] + 1.0, float(ko), "Goal · kick-off", "goal (from the goal event)"))
    # ball tracked off the pitch for 3 s+ with no provider stoppage: dead ball, estimated from the tracking
    t = 0
    while t < T:
        if seen[t] and outside(ball[t]) and not any(a - 4 <= t < b for a, b, _, _ in spells):
            u = t
            while u < T and seen[u] and outside(ball[u]):
                u += 1
            if u - t >= 3:
                spells.append((float(t), float(u), None, "estimated: tracked ball off the pitch"))
            t = u
        else:
            t += 1
    for s, e, kind, after in sorted(spells):
        si, ei = int(np.ceil(s)), min(T, int(np.floor(e)))
        for k in range(si - 1, max(0, si - 4) - 1, -1):  # the ball left the pitch just before play was stopped
            if seen[k] and outside(ball[k]):
                si = k
            elif seen[k]:
                break
        if ei - si < 1 or si >= T:
            continue
        stop = None
        for k in range(si, min(ei, si + 4)):
            if seen[k] and outside(ball[k]):
                stop = np.array([np.clip(ball[k][0], -1.5, L + 1.5), np.clip(ball[k][1], -1.5, W + 1.5)])
                break
        if stop is None:
            prev = [k for k in range(max(0, si - 4), si + 1) if seen[k]]
            stop = ball[prev[-1]].copy() if prev else ball[min(si, T - 1)].copy()
        if kind == "Goal · kick-off":
            g = next((x for x in goal_events if s - 15 <= x["timestamp"] < e), None)
            side = g["team_id"] if g else ("A" if stop[0] > L / 2 else "B")
            net = np.array([L + 1.2 if side == "A" else -1.2, np.clip(stop[1], W / 2 - 3.2, W / 2 + 3.2)])
            start = max(si, int(np.ceil(g["timestamp"])) + 1) if g else si  # in the net from the goal on, never before
            to_centre = max(start, ei - 2)  # one jump at the end of the stoppage: placed on the centre spot
            centre = np.array([L / 2, W / 2])
            end = ei
            tn = next((k for k in range(ei, min(T, ei + 10)) if seen[k]), None)
            if tn is not None and np.hypot(*(ball[tn] - centre)) > 25 * max(1, tn - ei + 1):
                # kicked off earlier than the provider's restart row: fly from the spot to where the ball is next tracked
                dur = int(np.ceil(np.hypot(*(ball[tn] - centre)) / 25))
                end = max(to_centre + 1, tn - dur)
                for k in range(end, tn):
                    a_ = (k - end) / max(1, tn - end)
                    b[k] = centre * (1 - a_) + ball[tn] * a_
            b[start:to_centre] = net
            b[to_centre:end] = centre
            for k in range(start, end):
                labels[k], restart_t[k] = kind, round(float(e), 1)
                if after.startswith("goal (from"):
                    basis[k] = "estimated_ball_state"
            continue
        # where and when play restarts: the TRACKED ball at rest and then kicked / thrown before the next possession
        # (the next possession is the RECEIVER, often far away); rule spots only when the ball is not tracked
        pi = min(T - 1, int(round(e + 1)))
        tr = [k for k in range(si + 1, min(T, pi + 1)) if seen[k]]
        r = None
        for i_, k in enumerate(tr[:-1]):
            k2 = tr[i_ + 1]
            if k2 - k > 3 or not (-2 <= ball[k][0] <= L + 2 and -2 <= ball[k][1] <= W + 2):
                continue
            at_rest = i_ == 0 or tr[i_ - 1] < k - 3 or np.hypot(*(ball[k] - ball[tr[i_ - 1]])) <= 3
            moved = np.hypot(*(ball[k2] - ball[k])) / (k2 - k)
            if at_rest and 3 <= moved <= 32:  # a physical kick / throw, not a tracking jump
                r = k  # keep the last such kick before the reception (ball still on the spot at second r)
        if r is not None:
            spot = ball[r].copy()
            flight = None
        else:
            near = [k for k in range(max(0, ei - 2), min(T, ei + 2)) if seen[k]]
            spot = ball[near[0]].copy() if near else stop.copy()
            left = stop[0] < L / 2
            if kind == "Throw-in":  # thrown in from the touchline where the ball went out
                side_y = -0.5 if stop[1] < W / 2 else W + 0.5
                spot = np.array([spot[0] if near and abs(spot[1] - side_y) < 6 else stop[0], side_y])
            elif kind == "Corner":
                spot = np.array([0.0 if left else L, 0.0 if stop[1] < W / 2 else W])
            elif kind == "Goal kick":
                spot = np.array([5.5 if left else L - 5.5, np.clip(spot[1], W / 2 - 9, W / 2 + 9)])
            elif kind == "Penalty":
                spot = np.array([11.0 if left else L - 11.0, W / 2])
            tn = next((k for k in range(pi, min(T, pi + 10)) if seen[k]), None)
            if tn is not None:  # the restart is kicked so that the ball arrives where it is next tracked (~25 m/s)
                dur = int(max(1, np.ceil(np.hypot(*(ball[tn] - spot)) / 25)))
                r = max(si + 1, tn - dur)
                flight = (r, tn)
            else:
                r, flight = ei - 1, None
        # r = the last second the ball sits on the restart spot; play restarts right after it
        r = int(min(max(r, si + 1), T - 1))
        hold = 1 if r - si >= 3 else 0
        b[si:si + hold] = stop                 # where play stopped
        b[si + hold:r + 1] = spot              # one jump: the ball is placed where play restarts
        if flight is not None:                 # restart taken: the ball flies on to where it is next tracked
            r0, tn = flight
            for k in range(r0, tn):
                a_ = (k - r0) / max(1, tn - r0)
                b[k] = spot * (1 - a_) + ball[tn] * a_
        for k in range(si, r + 1):
            labels[k], restart_t[k] = kind or "Dead ball", round(float(r + 1), 1)
            if after.startswith("estimated"):
                basis[k] = "estimated_ball_state"
    return b, labels, restart_t, basis


def glide_gaps(b, seen, labels):
    """Replay display only: across seconds where the ball is not tracked during live play, move it at constant speed
    from the last known to the next known position (it stays drawn as not tracked) instead of holding it and jumping
    when tracking resumes. Live views cannot use the next position and keep the last known one."""
    b = b.copy()
    T = len(b)
    anchor = np.array([bool(seen[t]) or labels[t] is not None for t in range(T)])
    t = 0
    while t < T:
        if anchor[t]:
            t += 1
            continue
        u = t
        while u < T and not anchor[u]:
            u += 1
        if t > 0 and u < T:
            for k in range(t, u):
                a = (k - t + 1) / (u - t + 1)
                b[k] = b[t - 1] * (1 - a) + b[u] * a
        t = u
    # a tracked step faster than any ball (> 32 m in one second) during live play is spread over the previous second
    for _ in range(2):
        for t in range(2, T):
            if labels[t] is None and labels[t - 1] is None and np.hypot(*(b[t] - b[t - 1])) > 32:
                b[t - 1] = (b[t - 2] + b[t]) / 2
    return b


def frames_for(clip, rec, poss_team, ball_seen):
    pos, r80, p5, vis = rec
    on = clip["on"]
    fr = []
    for t in range(len(pos)):
        xy, obs, conf, rad = [], [], [], []
        for j in range(pos.shape[1]):
            if not on[t, j] or not np.isfinite(pos[t, j]).all():
                xy.append(None), obs.append(None), conf.append(None), rad.append(None)
                continue
            o = bool(vis[t, j])
            p = clip["pos"][t, j] if o else pos[t, j]
            xy.append([round(float(p[0]), 2), round(float(p[1]), 2)])
            obs.append(o)
            conf.append(1.0 if o else round(float(p5[t, j]), 2))
            rad.append(0.0 if o else round(float(r80[t, j]), 1))
        fr.append({"t": t, "b": [round(float(clip["ball"][t, 0]), 2), round(float(clip["ball"][t, 1]), 2)],
                   "p": poss_team[t], "c": None, "xy": xy, "obs": obs, "conf": conf, "rad": rad,
                   "ball_observed": bool(ball_seen[t]), "bs": "tracked" if ball_seen[t] else "held_or_unknown"})
    return fr


def event_xy(x, y, attacking_side, flip, pl, pw):
    """Provider event coordinates are normalised so the event's team attacks left-to-right; undo that per event
    (checked: median 0.4-0.7 m to the tracked player), then apply the tracking orientation (home attacks +x)."""
    if x in ("", None) or y in ("", None):
        return None, None
    x, y = float(x), float(y)
    if attacking_side == "right_to_left":
        x, y = -x, -y
    px, py = x * L / pl + L / 2, y * W / pw + W / 2
    if flip:
        px, py = L - px, W - py
    return round(px, 1), round(py, 1)


GOAL_HALF_MOUTH = 3.66 + 0.8  # m: half the goal width plus tracking tolerance
LOOKBACK_S, LOOKAHEAD_S, ENTRY_WINDOW_S = 150, 20, 6


CROSSBAR_M = 2.44


def ball_entry(ball, ball_seen, side, t0, t1, ball_z=None):
    """First second in [t0, t1) where the TRACKED ball is over the goal line between the posts, under the crossbar
    (when its height is known), of the goal `side` attacks (home attacks +x). None when tracking never shows it."""
    for t in range(max(0, int(t0)), min(len(ball), int(np.ceil(t1)))):
        if not ball_seen[t]:
            continue
        x, y = ball[t]
        over = x >= L - 0.5 if side == "A" else x <= 0.5
        if not (over and abs(y - W / 2) <= GOAL_HALF_MOUTH):
            continue
        if ball_z is not None and np.isfinite(ball_z[t]) and ball_z[t] > CROSSBAR_M + 0.3:
            continue  # over the bar
        return t
    return None


def kickoff_after(ball, ball_seen, t, horizon=240):
    """Second at which the tracked ball is next on the centre spot (a real goal is followed by a kick-off)."""
    t0 = int(t) + 15
    return next((k for k in range(t0, min(len(ball), t0 + horizon)) if ball_seen[k]
                 and np.hypot(ball[k][0] - L / 2, ball[k][1] - W / 2) <= 3), None)


def attribute_goals(ev, goals, ball, ball_seen, goal_ends=None, ball_z=None):
    """Turn provider score changes into GOAL events, checked against the ball tracking.

    The provider's score fields can change before or long after the goal (~60-110 s, kick-off), so candidates are the
    scoring team's shots within [t_seen - 150 s, t_seen + 20 s]. A shot gets the goal only if the tracked ball enters
    that goal within 6 s and before the next candidate shot (a blocked shot followed by a scored rebound credits the
    rebound). Without such evidence: a direct ball-in-goal sighting (scorer unknown); else the provider's
    lead_to_goal shot / last shot, labelled unverified; else time unknown. Every GOAL says which evidence it rests on."""
    used, used_end, unlocated = set(), set(), []
    goal_ends = goal_ends or []  # provider: (time the scoring possession ended, scoring side, player id, own goal?)
    for side, t_seen, sc in goals:
        ends = [g for g in goal_ends if g[1] == side and id(g) not in used_end and t_seen - 200 <= g[0] <= t_seen + 20]
        if ends:
            # the provider marks the possession that ended in this goal: time it there (or at the tracked ball entry
            # just after it); the scorer is that possession's player unless it was an own goal
            g_end = max(ends, key=lambda g: g[0])
            used_end.add(id(g_end))
            te, _, pid, own, w0 = g_end
            approx = False
            if w0 is not None:
                # provider zero-length goal row (typical after set pieces, whose taker / scorer possessions are not
                # recorded): its time IS the goal (checked against tracking: the ball reaches the goal right then; the
                # set piece before it can take a minute to be taken) -> refine by the tracked ball entering the net
                # around it, else keep the row's time
                t_in = ball_entry(ball, ball_seen, side, te - 3, te + ENTRY_WINDOW_S, ball_z)
                if t_in is None:
                    t_in = float(te)
            else:
                t_in = ball_entry(ball, ball_seen, side, te - 2, te + ENTRY_WINDOW_S, ball_z)
            shot = None if w0 is not None else next((e for e in ev if e["event_type"] == "SHOT" and e["team_id"] == side
                                                     and id(e) not in used and abs(e["timestamp"] - te) <= 1.5), None)
            g = {"event_type": "GOAL", "team_id": side, "score": sc, "source": "skillcorner goal-ending possession",
                 "timestamp": float(t_in if t_in is not None else te),
                 "player_id": None if own else pid, "own_goal": bool(own),
                 "x": shot["x"] if shot else None, "y": shot["y"] if shot else None, "xg": shot["xg"] if shot else 0.0,
                 "evidence": ("provider goal row (set piece); ball not tracked into the goal" if approx else
                              "provider goal-ending possession; ball tracked entering the goal" if t_in is not None
                              else "provider goal-ending possession (ball not tracked into the goal)"),
                 **({"timing_approximate": True} if approx else {}),
                 "attribution": "provider own goal" if own else "provider goal-ending possession"}
            if shot is not None:
                used.add(id(shot))
                shot["outcome"], shot["on_target"] = "goal", True
                g["shot_timestamp"] = shot["timestamp"]
            ev.append(g)
            continue
        shots = sorted([e for e in ev if e["event_type"] == "SHOT" and e["team_id"] == side and id(e) not in used
                        and t_seen - LOOKBACK_S <= e["timestamp"] <= t_seen + LOOKAHEAD_S], key=lambda e: e["timestamp"])
        chosen, entry = None, None
        for k, s in enumerate(shots):
            nxt = shots[k + 1]["timestamp"] if k + 1 < len(shots) else s["timestamp"] + ENTRY_WINDOW_S
            t_in = ball_entry(ball, ball_seen, side, s["timestamp"], min(s["timestamp"] + ENTRY_WINDOW_S, max(nxt, s["timestamp"] + 1)), ball_z)
            if t_in is not None and kickoff_after(ball, ball_seen, t_in) is not None:  # tracking-only: needs the kick-off too
                chosen, entry = s, t_in  # keep the LAST shot whose ball goes in (rebounds after a save)
        g = {"event_type": "GOAL", "team_id": side, "score": sc, "source": "skillcorner score change"}
        if chosen is not None:
            used.add(id(chosen))
            chosen["outcome"], chosen["on_target"] = "goal", True
            g.update(timestamp=float(entry), player_id=chosen["player_id"], x=chosen["x"], y=chosen["y"], xg=chosen["xg"],
                     shot_timestamp=chosen["timestamp"], evidence="ball tracked entering the goal",
                     attribution="shot followed by the tracked ball entering the goal")
        else:
            t_in = ball_entry(ball, ball_seen, side, t_seen - LOOKBACK_S, t_seen + LOOKAHEAD_S, ball_z)
            if t_in is not None and kickoff_after(ball, ball_seen, t_in) is None:
                t_in = None
            if t_in is not None and not any(abs(e["timestamp"] - t_in) <= ENTRY_WINDOW_S for e in shots):
                g.update(timestamp=float(t_in), player_id=None, x=None, y=None, xg=0.0, evidence="ball tracked entering the goal",
                         attribution="no matching shot event; scorer unknown")
            else:
                # no provider goal-ending possession and no physically credible ball-in-net + kick-off: do NOT pin the
                # goal on a shot or a time (that produced false goal moments); it counts in the official score only
                unlocated.append({"team": side, "score_after": sc, "provider_score_change_s": round(float(t_seen), 1)})
                continue
        ev.append(g)
    return unlocated


def possession_changes(rows, pid_of, team_of, xy, ts, T, ball=None, ball_seen=None):
    """POSSESSION_CHANGE / KICK_OFF events from the provider's player-possession sequence (the engine's possession
    spells, recoveries, transitions and counters are bounded by these). Cause: the new team's on-ball engagement
    outcome (direct_regain -> tackle, indirect_regain -> interception) within 2 s of the change; a stoppage gap of
    3 s+ -> restart, or kickoff when play restarts on the centre spot; otherwise loose_ball."""
    poss = sorted([r for r in rows if r["event_type"] == "player_possession" and r["player_id"]
                   and int(r["player_id"]) in pid_of], key=lambda r: int(r["frame_start"]))
    regains = [r for r in rows if r["event_type"] == "on_ball_engagement" and r["end_type"] in ("direct_regain", "indirect_regain")
               and r["player_id"] and int(r["player_id"]) in team_of]
    out = []
    prev = None
    for r in poss:
        sid = int(r["player_id"])
        team = team_of[sid]
        t = ts(r["frame_start"])
        if t < 0 or t > T:
            prev = r
            continue
        x, y = xy(r["x_start"], r["y_start"], r)
        if x is None:
            prev = r
            continue
        on_spot = abs(x - L / 2) <= 4 and abs(y - W / 2) <= 4
        if prev is None:
            out.append({"timestamp": t, "event_type": "KICK_OFF", "team_id": team, "player_id": pid_of[sid], "x": x, "y": y,
                        "source": "first provider possession of the period"})
        elif team_of[int(prev["player_id"])] != team:
            f0, f1 = int(prev["frame_end"]), int(r["frame_start"])
            reg = [g for g in regains if team_of[int(g["player_id"])] == team and f0 - 20 <= int(g["frame_end"]) <= f1 + 20]
            gap = (f1 - f0) / 10
            before = r.get("game_interruption_before") or ""
            near_centre = True
            if ball is not None:  # a kick-off needs the tracked ball near the centre spot (provider labels can be off)
                ks = [k for k in range(max(0, int(t) - 3), min(len(ball), int(t) + 1)) if ball_seen[k]]
                near_centre = any(np.hypot(ball[k][0] - L / 2, ball[k][1] - W / 2) <= 8 for k in ks) if ks else True
            if before in ("goal_for", "goal_against") and near_centre:  # not "goal_kick_*"
                cause = "kickoff"
            elif before in ("goal_for", "goal_against"):
                cause = "restart"
            elif before:
                cause = "restart"
            elif reg:
                cause = "tackle" if any(g["end_type"] == "direct_regain" for g in reg) else "interception"
            elif gap >= 3 and on_spot:
                cause = "kickoff"
            elif gap >= 3:
                cause = "restart"
            else:
                cause = "loose_ball"
            out.append({"timestamp": t, "event_type": "POSSESSION_CHANGE", "team_id": team, "player_id": pid_of[sid],
                        "x": x, "y": y, "from_team": team_of[int(prev["player_id"])], "cause": cause,
                        "source": "skillcorner player possessions"})
            if cause == "kickoff" and any(o["event_type"] == "KICK_OFF" and t - o["timestamp"] < 60 for o in out):
                cause = out[-1]["cause"] = "restart"  # a second kick-off within a minute is a provider artefact
            if cause == "kickoff":
                out.append({"timestamp": max(0.0, t - 1.0), "event_type": "KICK_OFF", "team_id": team, "player_id": pid_of[sid],
                            "x": L / 2, "y": W / 2, "source": "provider kick-off after a goal / centre-spot restart"})
        prev = r
    return out


def build_events(mdir, period, clip, ids, pid_of, team_of, flip, pl, pw, f0, T, initial_score=None):
    """Provider dynamic events -> app events for one period (timestamps = seconds from the period's first frame)."""
    rows = list(csv.DictReader(open(glob.glob(os.path.join(mdir, "*_dynamic_events.csv"))[0])))
    rows = [r for r in rows if r["period"] == str(period)]

    def xy(x, y, r):
        return event_xy(x, y, r["attacking_side"], flip, pl, pw)

    def ts(frame):
        return round((int(frame) - f0) / 10, 1)

    ev = []
    for r in rows:
        sid = int(r["player_id"]) if r["player_id"] else None
        if sid not in pid_of:
            continue
        pid, team = pid_of[sid], team_of[sid]
        t = ts(r["frame_end"] or r["frame_start"])
        if t < 0 or t > T:
            continue
        if r["event_type"] == "player_possession":
            x, y = xy(r["x_end"], r["y_end"], r)
            if x is None:
                continue
            if r["end_type"] == "pass":
                tid = int(r["player_targeted_id"]) if r.get("player_targeted_id") else None
                tx, ty = xy(r.get("player_targeted_x_reception") or r.get("player_targeted_x_pass"),
                            r.get("player_targeted_y_reception") or r.get("player_targeted_y_pass"), r)
                if tx is None:
                    tx, ty = x, y
                sgn = 1 if team == "A" else -1
                ev.append({"timestamp": t, "event_type": "PASS", "team_id": team, "player_id": pid, "x": x, "y": y,
                           "target_x": tx, "target_y": ty, "receiver_id": pid_of.get(tid),
                           "success": r.get("pass_outcome") == "successful", "pressure": 0,
                           "distance": round(float(np.hypot(tx - x, ty - y)), 1), "progression": round(sgn * (tx - x), 1),
                           "cross": False, "source": "skillcorner_dynamic_events"})
            elif r["end_type"] == "shot":
                own = (x, y) if team == "A" else (L - x, W - y)
                ev.append({"timestamp": t, "event_type": "SHOT", "team_id": team, "player_id": pid, "x": x, "y": y,
                           "xg": round(_xg(own), 3), "outcome": "unknown", "on_target": None,
                           "distance": round(float(np.hypot(L - own[0], W / 2 - own[1])), 1),
                           "source": "skillcorner_dynamic_events", "xg_basis": "simulator distance/angle model",
                           "provider_lead_to_goal": r.get("lead_to_goal") == "True"})
            elif r["end_type"] in ("possession_loss",):
                pass
        elif r["event_type"] == "on_ball_engagement":
            x, y = xy(r["x_start"], r["y_start"], r)
            if x is None:
                continue
            tgt = int(r["player_in_possession_id"]) if r.get("player_in_possession_id") else None
            ev.append({"timestamp": ts(r["frame_start"]), "event_type": "PRESSURE", "team_id": team, "player_id": pid,
                       "x": x, "y": y, "target_player_id": pid_of.get(tgt), "pressure": 0.5,
                       "subtype": r["event_subtype"], "source": "skillcorner_dynamic_events"})
            if r["end_type"] in ("direct_regain",):
                ev.append({"timestamp": t, "event_type": "TACKLE", "team_id": team, "player_id": pid, "x": x, "y": y,
                           "opponent_id": pid_of.get(tgt), "success": True, "source": "skillcorner_dynamic_events"})
    ev += possession_changes(rows, pid_of, team_of, xy, ts, T, clip["ball"], clip["ball_seen"])
    # goals from score changes (team_score is the event team's score)
    score = dict(initial_score or {"A": 0, "B": 0})  # provider scores are cumulative over the match
    goals = []
    for r in sorted(rows, key=lambda r: int(r["frame_start"])):
        sid = int(r["player_id"]) if r["player_id"] else None
        if sid not in team_of or r["team_score"] == "":
            continue
        tm = team_of[sid]
        cur = {tm: int(r["team_score"]), ("B" if tm == "A" else "A"): int(r["opponent_team_score"])}
        for side in ("A", "B"):
            while cur[side] > score[side]:
                score[side] += 1
                goals.append((side, ts(r["frame_start"]), dict(score)))
    goal_ends = []
    poss_rows = sorted([r for r in rows if r["event_type"] == "player_possession"], key=lambda r: int(r["frame_start"]))
    for i, r in enumerate(poss_rows):
        lab = r.get("game_interruption_after") or ""
        if lab in ("goal_for", "goal_against") and r["player_id"] and int(r["player_id"]) in team_of:
            tm = team_of[int(r["player_id"])]
            own = lab == "goal_against"
            placeholder = r["frame_start"] == r["frame_end"] and (r.get("game_interruption_before") or "") in ("goal_for", "goal_against")
            w0 = ts(poss_rows[i - 1]["frame_end"]) + 1.0 if placeholder and i > 0 else None
            goal_ends.append((ts(r["frame_end"]), ("B" if tm == "A" else "A") if own else tm, pid_of[int(r["player_id"])], own, w0))
    unlocated = attribute_goals(ev, goals, clip["ball"], clip["ball_seen"], goal_ends, clip.get("ball_z"))
    counted = [e["timestamp"] for e in ev if e["event_type"] == "GOAL"]
    for e in list(ev):  # a provider "restart after a goal" label with no counted goal before it is not a kick-off
        if e["event_type"] == "KICK_OFF" and e["timestamp"] > 15 and not any(0 < e["timestamp"] - g < 300 for g in counted):
            ev.remove(e)
        elif e["event_type"] == "POSSESSION_CHANGE" and e.get("cause") == "kickoff" \
                and not any(0 < e["timestamp"] - g < 300 for g in counted):
            e["cause"] = "restart"
    return ev, score, unlocated


def export(mdir, models, model_name, out_dir=DATA):
    mid = os.path.basename(mdir)
    meta = json.load(open(glob.glob(os.path.join(mdir, "*_match.json"))[0]))
    pl, pw = float(meta.get("pitch_length") or 105), float(meta.get("pitch_width") or 68)
    home = meta["home_team"]["id"]
    sides = meta.get("home_team_side") or ["left_to_right", "right_to_left"]
    players = {p["id"]: p for p in meta["players"]}
    periods = {p["period"]: p for p in meta["match_periods"]}
    files, score0, placed = [], {"A": 0, "B": 0}, {"A": 0, "B": 0}
    for name, clip in load_match(mdir):
        period = int(name.rsplit("_p", 1)[1])
        flip = sides[period - 1] == "right_to_left"
        T, N = clip["pos"].shape[:2]
        # same player-id order as load_match
        ids = sorted({p for p in _period_ids(mdir, period)})
        assert len(ids) == N
        team_of, pid_of, plist, order = {}, {}, [], []
        for sid, tm in zip(ids, clip["team"]):
            p = players[sid]
            app_id = f"{tm}{p['number']}"
            if app_id in order:
                app_id = f"{tm}{p['number']}_{sid}"
            team_of[sid], pid_of[sid] = tm, app_id
            order.append(app_id)
            acr = p["player_role"]["acronym"]
            plist.append({"id": app_id, "team": tm, "shirt": p["number"], "name": p.get("short_name") or p["last_name"],
                          "position": POSITION.get(acr, "MID") if acr != "GK" else "GK", "start_role": POSITION.get(acr, "MID"),
                          "role_slot": acr, "provider_id": sid})
        f0 = int(periods[period]["start_frame"])
        f0 = f0 + (-f0) % 10  # first sampled frame (frame % 10 == 0)
        ev, score, unlocated = build_events(mdir, period, clip, ids, pid_of, team_of, flip, pl, pw, f0, T, score0)
        for e in ev:
            if e["event_type"] == "GOAL":
                placed[e["team_id"]] += 1
        # substitutions from the match sheet: player on = starts after the period start; player off = same team,
        # leaves within 10 s of that frame
        spans = {}
        for sid, p in players.items():
            for bp in (p.get("playing_time") or {}).get("by_period") or []:
                if bp["name"] == f"period_{period}" and sid in pid_of:
                    spans[sid] = (int(bp["start_frame"]), int(bp["end_frame"]))
        used_off = set()
        for sid, (a, b) in sorted(spans.items(), key=lambda x: x[1][0]):
            if (a - f0) / 10 <= 5:
                continue
            # one-to-one: the not-yet-replaced team-mate whose playing time ends closest to this player's start
            off = sorted([o for o, (a2, b2) in spans.items() if o != sid and o not in used_off and team_of[o] == team_of[sid]
                          and abs(b2 - a) <= 100], key=lambda o: abs(spans[o][1] - a))
            if not off:
                continue
            used_off.add(off[0])
            ev.append({"timestamp": round((a - f0) / 10, 1), "event_type": "SUBSTITUTION", "team_id": team_of[sid],
                       "player_id": pid_of[sid], "player_on_id": pid_of[sid], "player_off_id": pid_of[off[0]],
                       "role": players[sid]["player_role"]["acronym"], "x": None, "y": None, "source": "skillcorner match sheet"})
        ev.sort(key=lambda e: e["timestamp"])
        for k, e in enumerate(ev):
            e["id"] = f"e{k:05d}"
        # coordinate sanity: provider event location vs that player's tracked detection at the same second
        dist = []
        for e in ev:
            if e["event_type"] in ("PASS", "SHOT") and e["player_id"] in order:
                t, j = int(round(e["timestamp"])), order.index(e["player_id"])
                if 0 <= t < T and np.isfinite(clip["pos"][t, j]).all():
                    dist.append(float(np.hypot(*(clip["pos"][t, j] - [e["x"], e["y"]]))))
        recs = reconstruct(clip, models)
        ball_seen = clip["ball_seen"]
        poss_team = [None if p < 0 else ("A" if p == 0 else "B") for p in clip["poss"]]
        replay = frames_for(clip, recs["replay"], poss_team, ball_seen)
        live = frames_for(clip, recs["live"], poss_team, ball_seen)
        shown, dead, restart_at, dead_basis = display_ball(clip, [e for e in ev if e["event_type"] == "GOAL"])
        glided = glide_gaps(shown, clip["ball_seen"], dead)
        for fr_list, retro in ((replay, True), (live, False)):
            for t, f in enumerate(fr_list):
                pos = glided[t] if retro else shown[t]
                f["b"] = [round(float(pos[0]), 2), round(float(pos[1]), 2)]
                if dead[t]:
                    f.update(ds=dead[t], ds_basis=dead_basis[t], bs="dead_ball")
                    if retro:  # when play restarts is only known afterwards
                        f["ds_restart_t"] = restart_at[t]
        vis = recs["replay"][3]
        teams = {"A": meta["home_team"], "B": meta["away_team"]}
        final = {"A": meta["home_team_score"], "B": meta["away_team_score"]} if period == 2 else score
        # official goals that no evidence locates in time (whole match so far; shown on the lobby card)
        missing = {k: final[k] - placed[k] for k in final} if period == 2 else {k: len([u for u in unlocated if u["team"] == k]) for k in final}
        data = {
            "meta": {"seed": None, "view": "cv", "duration": T - 1, "clock_offset": 0 if period == 1 else 45 * 60,
                     "pitch": {"length": L, "width": W},
                     "teams": {k: {"name": v["name"], "short": v.get("short_name", v["name"])[:3].upper(), "color": COLORS[i]}
                               for i, (k, v) in enumerate(teams.items())},
                     "player_order": order, "initial_score": score0, "final_score": final, "score_known": True,
                     "date": (meta.get("date_time") or "")[:10],
                     "source": {"type": "computer_vision", "provider": "SkillCorner", "match_id": int(mid), "half": period,
                                "game": f"skillcorner/{mid}", "competition": "A-League 2024/25",
                                "dataset": "SkillCorner Open Data (MIT), broadcast tracking",
                                "date": (meta.get("date_time") or "")[:10],
                                "observations": "SkillCorner broadcast detections (is_detected) at 1 Hz; provider extrapolations discarded",
                                "observed_share": round(float(vis.sum() / max(1, clip["on"].sum())), 3),
                                "ball_observed_share": round(float(ball_seen.mean()), 3),
                                "seconds_without_provider_tracking": int((~clip["tracked"]).sum()),
                                "on_pitch_basis": "match-sheet playing time (provider lists nobody during replays / close-ups; players are estimated there)",
                                "reconstruction_model": model_name, "reconstruction_trained_on": "DFL (7 Bundesliga matches); never on this match",
                                "retrospective": True, "mode": "replay", "live_available": True,
                                "ground_truth_available": False,
                                "event_feed": "SkillCorner dynamic events (provider CV-derived); goals from score changes; xG from simulator model",
                                "unavailable_stats": ["shots_on_target", "fouls", "corners", "offsides", "yellow_cards", "red_cards"],
                                "event_location_vs_tracking_median_m": round(float(np.median(dist)), 2) if dist else None,
                                "uncertainty": "DFL-calibrated 80% radius; broadcast-domain coverage measured on GSR/SkillCorner re-entries, see docs/reconstruction_evaluation.md",
                                "uncertainty_validated": {"basis": "SkillCorner A-League re-entries (proxy truth = provider detections), model r13_mlp_aug_ens3_final",
                                                          "live_r80_coverage": 0.776, "replay_r80_coverage": 0.807,
                                                          "live_p5_calibration": "predicted vs observed within 0.03 in every bin",
                                                          "applies_to": "estimated players only; measured positions remain unverified"},
                                "orientation_basis": "provider home_team_side per period",
                                "goals_not_in_provider_events": missing if any(missing.values()) else None,
                                "goals_not_located": unlocated or None,
                                "warning": "Hidden players are model estimates, not observations. Identity and positions of observed players are SkillCorner's, not independently verified."}},
            "players": plist, "events": ev, "frames": replay, "alt_frames": {"live": live, "replay": replay}}
        score0 = dict(final)
        path = out_dir / f"skc_{mid}_h{period}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, separators=(",", ":")))
        tmp.replace(path)
        files.append({"file": str(path.relative_to(ROOT)), "seconds": T, "players": N, "events": len(ev),
                      "goals": sum(e["event_type"] == "GOAL" for e in ev), "observed_share": data["meta"]["source"]["observed_share"],
                      "event_location_vs_tracking_median_m": data["meta"]["source"]["event_location_vs_tracking_median_m"]})
    return files


def _period_ids(mdir, period):
    ids = set()
    with open(glob.glob(os.path.join(mdir, "*tracking_extrapolated.jsonl"))[0]) as fh:
        for line in fh:
            fr = json.loads(line)
            if fr.get("period") == period and fr["frame"] % 10 == 0:
                ids.update(p["player_id"] for p in fr["player_data"])
    return ids


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="e.g. r13_mlp_final or final_v3 (see gsr_benchmark.model_path)")
    ap.add_argument("--match", default=None)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    models = {mode: hm.load_hidden(model_path(a.model, mode)) for mode in ("live", "replay")}
    dirs = sorted(d for d in glob.glob(str(SC_DIR / "data" / "matches" / "*")) if os.path.isdir(d))
    if a.match:
        dirs = [d for d in dirs if os.path.basename(d) == a.match]
    log = []
    for d in dirs:
        r = export(d, models, a.model)
        print(r, flush=True)
        log += r
    (DATA / "eval" / "skillcorner_export_log.json").write_text(json.dumps(log, indent=1))
