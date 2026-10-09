from typing import List

from backend.models.match import MatchData, to_own


def possession_sequences(m: MatchData) -> List[dict]:
    """Contiguous spells of possession, bounded by POSSESSION_CHANGE / KICK_OFF events."""
    starts = [e for e in m.events if e["event_type"] in ("POSSESSION_CHANGE", "KICK_OFF")]
    seqs = []
    for i, s in enumerate(starts):
        end = starts[i + 1]["timestamp"] if i + 1 < len(starts) else m.duration
        team = s["team_id"]
        evs = [e for e in m.events if s["timestamp"] <= e["timestamp"] <= end and e["team_id"] == team
               and e["event_type"] in ("PASS", "SHOT", "GOAL")]
        # don't include events of the same second that belong to the previous spell
        evs = [e for e in evs if e["id"] > s["id"]]
        t0, t1 = int(s["timestamp"]), int(end)
        bx = m.own_ball(team)[t0:t1 + 1, 0]
        sx, sy = to_own(team, s.get("x", 52.5), s.get("y", 34))
        seqs.append({
            "team": team, "start": s["timestamp"], "end": end, "start_event": s["id"],
            "cause": s.get("cause", "kickoff"), "player_id": s["player_id"],
            "start_own_x": round(sx, 1), "start_own_y": round(sy, 1),
            "max_own_x": round(float(bx.max()) if len(bx) else sx, 1),
            "ball_own_x": [round(float(v), 1) for v in bx],
            "passes": [e for e in evs if e["event_type"] == "PASS"],
            "shots": [e for e in evs if e["event_type"] == "SHOT"],
            "goal": next((e for e in evs if e["event_type"] == "GOAL"), None),
        })
    return seqs
