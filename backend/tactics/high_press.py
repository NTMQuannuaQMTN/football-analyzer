"""High press detection + tactic -> outcome tracing."""
from typing import Dict, List

from backend.analytics.passing import is_backward
from backend.models.match import MatchData, other, to_own

MIN_MINUTES = 4


def detect_high_press(m: MatchData, timeline: Dict[str, List[dict]], seqs: List[dict]) -> List[dict]:
    out = []
    for team in ("A", "B"):
        rows = timeline[team]
        flags = [r.get("high_press", False) for r in rows]  # see tactical_state.high_press_flags
        # close single-minute gaps
        for k in range(1, len(flags) - 1):
            if not flags[k] and flags[k - 1] and flags[k + 1]:
                flags[k] = True
        spans, k = [], 0
        while k < len(flags):
            if not flags[k]:
                k += 1
                continue
            j = k
            while j < len(flags) and flags[j]:
                j += 1
            if j - k >= MIN_MINUTES:
                # smoothed metrics lag 1-2 minutes: walk back to the real onset
                onset = k
                while onset > 0 and rows[onset - 1]["press_index"] >= 50:
                    onset -= 1
                if spans and onset - spans[-1][1] <= 3:
                    spans[-1][1] = j
                else:
                    spans.append([onset, j])
            k = j
        out += [_segment(m, team, a, b, timeline, seqs) for a, b in spans]
    return out


def _segment(m, team, m0, m1, timeline, seqs):
    t0, t1 = m0 * 60, m1 * 60
    opp = other(team)
    rec = [e for e in m.events_between(t0, t1, ("POSSESSION_CHANGE",), team)
           if e.get("cause") in ("tackle", "interception")]
    chains = []
    for e in rec:
        ox = to_own(team, e["x"], e["y"])[0]
        if ox < 70:
            continue
        seq = next((s for s in seqs if s["start_event"] == e["id"]), None)
        # what the opponent did just before losing it
        before = [p for p in m.events_between(e["timestamp"] - 12, e["timestamp"] + 0.1, ("PASS",), opp)]
        backward = sum(1 for p in before if is_backward(p))
        shot = seq["shots"][0] if seq and seq["shots"] else None
        chains.append({
            "recovery_event": e["id"], "timestamp": e["timestamp"], "player_id": e["player_id"],
            "own_x": round(ox, 1), "opp_backward_passes_before": backward,
            "shot": shot["id"] if shot else None,
            "seconds_to_shot": round(shot["timestamp"] - e["timestamp"], 1) if shot else None,
            "xg": shot["xg"] if shot else 0,
            "goal": seq["goal"]["id"] if seq and seq["goal"] else None,
        })
    # press bypassed: opponent spells starting deep that reach our box zone quickly
    bypasses = []
    for s in seqs:
        if s["team"] != opp or not (t0 <= s["start"] < t1):
            continue
        bx = s["ball_own_x"]
        if bx and min(bx[:3]) < 35:
            reach = next((i for i, v in enumerate(bx) if v >= 70), None)
            if reach is not None and reach <= 15 and len([p for p in s["passes"] if p["timestamp"] <= s["start"] + reach]) <= 4:
                bypasses.append({"start": s["start"], "seconds_to_final_third": reach,
                                 "shot": s["shots"][0]["id"] if s["shots"] else None,
                                 "xg": s["shots"][0]["xg"] if s["shots"] else 0,
                                 "goal": s["goal"]["id"] if s["goal"] else None})
    return {"type": "HIGH_PRESS", "team": team, "start": t0, "end": t1, "start_min": m0, "end_min": m1,
            "peak_press_index": max(r["press_index"] for r in timeline[team][m0:m1]),
            "final_third_recovery_chains": chains, "bypasses": bypasses}
