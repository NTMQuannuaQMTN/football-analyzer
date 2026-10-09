from backend.models.match import MatchData, to_own


def is_progressive(e) -> bool:
    return bool(e.get("success")) and e.get("progression", 0) >= 10


def is_backward(e) -> bool:
    return e.get("progression", 0) <= -3


def pass_stats(passes) -> dict:
    n = len(passes)
    ok = sum(1 for e in passes if e["success"])
    return {"passes": n, "completed": ok,
            "accuracy": round(ok / n, 3) if n else None,
            "avg_distance": round(sum(e["distance"] for e in passes) / n, 1) if n else None,
            "progressive": sum(1 for e in passes if is_progressive(e)),
            "backward": sum(1 for e in passes if is_backward(e))}


def player_event_stats(m: MatchData, t_end: float = None) -> dict:
    t_end = m.duration + 1 if t_end is None else t_end
    out = {pid: {"passes": 0, "completed": 0, "progressive_passes": 0, "pressures": 0,
                 "tackles": 0, "recoveries": 0, "final_third_recoveries": 0, "shots": 0, "xg": 0.0,
                 "goals": 0, "assists": 0, "right_side_actions": 0} for pid in m.order}
    for e in m.events:
        if e["timestamp"] > t_end:
            break
        pid, et = e["player_id"], e["event_type"]
        if pid not in out:
            continue
        s = out[pid]
        if et == "PASS":
            s["passes"] += 1
            s["completed"] += int(e["success"])
            s["progressive_passes"] += int(is_progressive(e))
        elif et == "PRESSURE":
            s["pressures"] += 1
            ox, oy = to_own(e["team_id"], e["x"], e["y"])
            s["right_side_actions"] += int(oy > 44)
        elif et == "TACKLE":
            s["tackles"] += 1
        elif et == "POSSESSION_CHANGE" and e.get("cause") in ("tackle", "interception"):
            s["recoveries"] += 1
            ox, _ = to_own(e["team_id"], e["x"], e["y"])
            s["final_third_recoveries"] += int(ox >= 70)
        elif et == "SHOT":
            s["shots"] += 1
            s["xg"] = round(s["xg"] + e["xg"], 3)
        elif et == "GOAL":
            s["goals"] += 1
            if e.get("assist_id") in out:
                out[e["assist_id"]]["assists"] += 1
    return out
