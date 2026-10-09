"""Counterattack detection: a temporal sequence, not a single threshold.

  1. RECOVERY       possession won by a tackle or interception in the own half or midfield
  2. DISORGANISED   at the recovery few opponents are goal-side of the ball (a settled defence has 8-10 there)
  3. PROGRESSION    the ball travels >= PROG_M forward, fast (>= SPEED_MS on average), with few passes
  4. ENTRY          it reaches the final third within WINDOW_S seconds (or a shot comes first)

What it is not: a fast attack against a settled defence (step 2 fails), sustained possession (too slow / too many
passes), or any attacking transition that does not progress (steps 3-4 fail).
Designed on 5 DFL development matches (labels: DFL's CounterAttack flag on shots) and the synthetic benchmark;
the two Bundesliga matches are kept as a held-out check (backend/tactics/real_eval.py).
"""
from typing import List

import numpy as np

from backend.models.match import MatchData

WINDOW_S = 15
MAX_RECOVERY_X = 65     # own-frame x of the recovery (m from own goal line)
FINAL_THIRD_X = 70
PROG_M = 25
SPEED_MS = 3.5
MAX_PASSES = 4
MAX_GOAL_SIDE = 7       # opponents (outfield) between the ball and their own goal at the recovery
CAUSES = ("tackle", "interception")  # loose-ball starts doubled false positives on the dev matches


def opponents_goal_side(m: MatchData, team: str, t: int):
    """Outfield opponents nearer their own goal than the ball (team's own frame), at second t."""
    opp = "B" if team == "A" else "A"
    xs = m.own_xy(team)[t][m.team_mask(opp)][:, 0]
    bx = m.own_ball(team)[t, 0]
    xs = xs[np.isfinite(xs)]
    return int((xs > bx).sum()), len(xs)


def detect_counterattacks(m: MatchData, seqs: List[dict]) -> List[dict]:
    out = []
    for s in seqs:
        if s["cause"] not in CAUSES or s["start_own_x"] > MAX_RECOVERY_X:
            continue
        t0 = int(s["start"])
        goal_side, seen = opponents_goal_side(m, s["team"], min(t0, m.T - 1))
        if seen < 7 or goal_side > MAX_GOAL_SIDE:
            continue  # settled defence (or too few opponents visible to judge)
        bx = s["ball_own_x"][:WINDOW_S + 1]
        shot = s["shots"][0] if s["shots"] else None
        reach = next((i for i, v in enumerate(bx) if v >= FINAL_THIRD_X), None)
        if reach is None and shot is not None and shot["timestamp"] - s["start"] <= WINDOW_S:
            reach = max(1, int(round(shot["timestamp"] - s["start"])))
        if reach is None or reach < 2:
            continue
        progressed = max(bx[:reach + 1]) - bx[0]
        passes = [p for p in s["passes"] if p["timestamp"] <= s["start"] + reach]
        speed = progressed / reach
        if progressed < PROG_M or len(passes) > MAX_PASSES or speed < SPEED_MS:
            continue
        outcome = "goal" if s["goal"] else ("shot" if shot else "lost")
        out.append({
            "type": "COUNTERATTACK", "team": s["team"], "start": s["start"],
            "end": shot["timestamp"] if shot else s["start"] + reach,
            "recovery_player": s["player_id"], "recovery_own_x": s["start_own_x"],
            "opponents_goal_side": goal_side,
            "progression_m": round(progressed, 1), "seconds": reach, "passes": len(passes),
            "speed_ms": round(speed, 1), "outcome": outcome,
            "stages": ["recovery", "disorganised_defence", "rapid_progression", "final_third_entry"]
                      + (["shot"] if shot else []),
            "shot": shot["id"] if shot else None, "xg": shot["xg"] if shot else 0,
            "goal": s["goal"]["id"] if s["goal"] else None,
            "players": sorted({s["player_id"], *[p["player_id"] for p in passes],
                               *[p["receiver_id"] for p in passes if p.get("receiver_id")],
                               *([shot["player_id"]] if shot else [])}),
        })
    return out
