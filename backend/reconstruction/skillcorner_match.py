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


def ball_entry(ball, ball_seen, side, t0, t1):
    """First second in [t0, t1) where the TRACKED (not forward-filled) ball is over the goal line, between the posts,
    of the goal `side` attacks (home attacks +x). None when tracking never shows it."""
    for t in range(max(0, int(t0)), min(len(ball), int(np.ceil(t1)))):
        if not ball_seen[t]:
            continue
        x, y = ball[t]
        over = x >= L - 0.5 if side == "A" else x <= 0.5
        if over and abs(y - W / 2) <= GOAL_HALF_MOUTH:
            return t
    return None


def attribute_goals(ev, goals, ball, ball_seen):
    """Turn provider score changes into GOAL events, checked against the ball tracking.

    The provider's score fields can change before or long after the goal (~60-110 s, kick-off), so candidates are the
    scoring team's shots within [t_seen - 150 s, t_seen + 20 s]. A shot gets the goal only if the tracked ball enters
    that goal within 6 s and before the next candidate shot (a blocked shot followed by a scored rebound credits the
    rebound). Without such evidence: a direct ball-in-goal sighting (scorer unknown); else the provider's
    lead_to_goal shot / last shot, labelled unverified; else time unknown. Every GOAL says which evidence it rests on."""
    used = set()
    for side, t_seen, sc in goals:
        shots = sorted([e for e in ev if e["event_type"] == "SHOT" and e["team_id"] == side and id(e) not in used
                        and t_seen - LOOKBACK_S <= e["timestamp"] <= t_seen + LOOKAHEAD_S], key=lambda e: e["timestamp"])
        chosen, entry = None, None
        for k, s in enumerate(shots):
            nxt = shots[k + 1]["timestamp"] if k + 1 < len(shots) else s["timestamp"] + ENTRY_WINDOW_S
            t_in = ball_entry(ball, ball_seen, side, s["timestamp"], min(s["timestamp"] + ENTRY_WINDOW_S, max(nxt, s["timestamp"] + 1)))
            if t_in is not None:
                chosen, entry = s, t_in  # keep the LAST shot whose ball goes in (rebounds after a save)
        g = {"event_type": "GOAL", "team_id": side, "score": sc, "source": "skillcorner score change"}
        if chosen is not None:
            used.add(id(chosen))
            chosen["outcome"], chosen["on_target"] = "goal", True
            g.update(timestamp=float(entry), player_id=chosen["player_id"], x=chosen["x"], y=chosen["y"], xg=chosen["xg"],
                     shot_timestamp=chosen["timestamp"], evidence="ball tracked entering the goal",
                     attribution="shot followed by the tracked ball entering the goal")
        else:
            t_in = ball_entry(ball, ball_seen, side, t_seen - LOOKBACK_S, t_seen + LOOKAHEAD_S)
            flagged = [e for e in shots if e.get("provider_lead_to_goal")]
            if t_in is not None and not any(abs(e["timestamp"] - t_in) <= ENTRY_WINDOW_S for e in shots):
                g.update(timestamp=float(t_in), player_id=None, x=None, y=None, xg=0.0, evidence="ball tracked entering the goal",
                         attribution="no matching shot event; scorer unknown")
            elif shots:
                s = (flagged or shots)[-1]
                used.add(id(s))
                s["outcome"], s["on_target"] = "goal", True
                g.update(timestamp=s["timestamp"], player_id=s["player_id"], x=s["x"], y=s["y"], xg=s["xg"],
                         evidence="ball NOT seen entering the goal (tracking lost); timing and scorer unverified",
                         attribution="provider lead_to_goal shot" if flagged else "last shot before the provider score change",
                         unverified=True)
            else:
                g.update(timestamp=max(0.0, t_seen - 1), player_id=None, x=None, y=None, xg=0.0, timing_approximate=True,
                         evidence="no shot and no tracked ball entry found; time and scorer unknown", unverified=True,
                         attribution="none")
        ev.append(g)
    return ev


def possession_changes(rows, pid_of, team_of, xy, ts, T):
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
            if reg:
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
            if cause == "kickoff":
                out.append({"timestamp": t, "event_type": "KICK_OFF", "team_id": team, "player_id": pid_of[sid], "x": x, "y": y,
                            "source": "provider possession restarting on the centre spot"})
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
    ev += possession_changes(rows, pid_of, team_of, xy, ts, T)
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
    attribute_goals(ev, goals, clip["ball"], clip["ball_seen"])
    return ev, score


def export(mdir, models, model_name, out_dir=DATA):
    mid = os.path.basename(mdir)
    meta = json.load(open(glob.glob(os.path.join(mdir, "*_match.json"))[0]))
    pl, pw = float(meta.get("pitch_length") or 105), float(meta.get("pitch_width") or 68)
    home = meta["home_team"]["id"]
    sides = meta.get("home_team_side") or ["left_to_right", "right_to_left"]
    players = {p["id"]: p for p in meta["players"]}
    periods = {p["period"]: p for p in meta["match_periods"]}
    files, score0 = [], {"A": 0, "B": 0}
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
        ev, score = build_events(mdir, period, clip, ids, pid_of, team_of, flip, pl, pw, f0, T, score0)
        # substitutions from the match sheet: player on = starts after the period start; player off = same team,
        # leaves within 10 s of that frame
        spans = {}
        for sid, p in players.items():
            for bp in (p.get("playing_time") or {}).get("by_period") or []:
                if bp["name"] == f"period_{period}" and sid in pid_of:
                    spans[sid] = (int(bp["start_frame"]), int(bp["end_frame"]))
        for sid, (a, b) in spans.items():
            if (a - f0) / 10 <= 5:
                continue
            off = [o for o, (a2, b2) in spans.items() if o != sid and team_of[o] == team_of[sid] and abs(b2 - a) <= 100]
            if not off:
                continue
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
        vis = recs["replay"][3]
        teams = {"A": meta["home_team"], "B": meta["away_team"]}
        final = {"A": meta["home_team_score"], "B": meta["away_team_score"]} if period == 2 else score
        missing = {k: final[k] - score[k] for k in final} if period == 2 else {"A": 0, "B": 0}
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
