"""How football-like is the simulator? The same behaviour statistics on simulated and real (DFL) matches.

  backward_pass_share      passes going >= 3 m backwards
  final_third_back_share   of passes played from the final third, share going backwards
  passes_per_min, shots_per_match, shot_distance_m (median)
  through_on_goal          seconds where the carrier is in the attacking half with no outfield defender goal-side
                           inside a 12 m corridor to goal; what he did next (shot / carry on / pass back / pass other / lost)

    .venv/bin/python -m simulator.realism
"""
import glob
import json

import numpy as np

from backend.models.match import L, W, MatchData, to_own


def stats(m: MatchData):
    ev = m.events
    passes = [e for e in ev if e["event_type"] == "PASS"]
    shots = [e for e in ev if e["event_type"] == "SHOT"]
    back = [p for p in passes if (p.get("progression") or 0) <= -3]
    ft = [p for p in passes if to_own(p["team_id"], p["x"], p["y"])[0] >= 70]
    ft_back = [p for p in ft if (p.get("progression") or 0) <= -3]
    out = {"passes_per_min": round(len(passes) / (m.duration / 60), 2),
           "backward_pass_share": round(len(back) / max(1, len(passes)), 3),
           "final_third_back_share": round(len(ft_back) / max(1, len(ft)), 3),
           "shots_per_90": round(len(shots) / (m.duration / 5400), 1),
           "shot_distance_m": round(float(np.median([np.hypot(L - to_own(s["team_id"], s["x"], s["y"])[0],
                                                              to_own(s["team_id"], s["x"], s["y"])[1] - W / 2) for s in shots])), 1) if shots else None}
    # through on goal: carrier, attacking half, no outfield defender between him and goal in a 12 m corridor
    nxt = {"shot": 0, "carry_on": 0, "pass_back": 0, "pass_other": 0, "lost": 0}
    n = 0
    seen_until = -1
    for t in range(m.T):
        f = m.frames[t]
        c, tm = f.get("c"), f.get("p")
        if not c or not tm or t <= seen_until:
            continue
        j = m.idx.get(c)
        if j is None or not np.isfinite(m.xy[t, j]).all():
            continue
        cx, cy = to_own(tm, *m.xy[t, j])
        if cx < 70:
            continue
        opp_mask = m.team_mask("B" if tm == "A" else "A")
        ox = m.own_xy(tm)[t][opp_mask]
        goal_side = [(x, y) for x, y in ox if np.isfinite(x) and x > cx and abs(y - (cy + (W / 2 - cy) * (x - cx) / max(1, L - cx))) < 6]
        if goal_side:
            continue
        n += 1
        seen_until = t + 5
        acts = [e for e in m.events if t <= e["timestamp"] <= t + 5 and e["event_type"] in ("SHOT", "PASS", "POSSESSION_CHANGE")]
        a = acts[0] if acts else None
        if a is None:
            nxt["carry_on"] += 1
        elif a["event_type"] == "SHOT":
            nxt["shot"] += 1
        elif a["event_type"] == "POSSESSION_CHANGE":
            nxt["lost"] += 1
        elif (a.get("progression") or 0) <= -3:
            nxt["pass_back"] += 1
        else:
            nxt["pass_other"] += 1
    out["through_on_goal"] = {"n": n, **{k: round(v / max(1, n), 2) for k, v in nxt.items()}}
    return out


def mean(rows):
    keys = [k for k in rows[0] if k != "through_on_goal"]
    out = {k: round(float(np.mean([r[k] for r in rows if r[k] is not None])), 3) for k in keys}
    tog = [r["through_on_goal"] for r in rows]
    out["through_on_goal"] = {k: round(float(np.mean([x[k] for x in tog])), 2) for k in tog[0]}
    return out


def main(seeds=(7, 11, 23)):
    from simulator.match_generator import generate_match
    sim = mean([stats(MatchData(generate_match(s))) for s in seeds])
    real = []
    for f in sorted(glob.glob("data/dfl_*_h[12].json")):
        raw = json.load(open(f))
        raw.pop("alt_frames", None)
        real.append(stats(MatchData(raw)))
    real = mean(real)
    for k in sim:
        print(f"{k:24s} simulator {sim[k]}   real (DFL) {real[k]}")
    return sim, real


if __name__ == "__main__":
    main()
