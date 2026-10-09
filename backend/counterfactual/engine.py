"""What-if engine. MVP: re-simulate the controlled scenario with one tactical decision removed
(same seed) and run the identical analytics over the same window. A learned simulation model
can replace `simulate` later without changing the output contract."""
from functools import lru_cache

from backend.models.match import MatchData
from backend.tactics.tactical_state import build_timeline, window_metrics
from simulator.match_generator import generate_match
from simulator.scenarios import ALL_FLAGS

SCENARIOS = {
    # moment selector -> (flags removed, question)
    "high_press": (("A_press",), "What if Ashford had not pressed high?"),
    "low_block": (("B_counter",), "What if Belmont had not dropped into a low block?"),
    "FORMATION_SHIFT": (("A_shift",), "What if Ashford had stayed in 4-3-3?"),
    "SUBSTITUTION_IMPACT": (("A_sub",), "What if the substitution had not been made?"),
    "RED_CARD_IMPACT": (("B_red",), "What if Belmont had kept 11 players?"),
}

ROWS = [("press_intensity", "Press intensity", ""), ("territory_pct", "Territory", "%"),
        ("possession_pct", "Possession", "%"), ("final_third_recoveries", "Final-third recoveries", ""),
        ("shots", "Shots", ""), ("xg", "xG", ""), ("goals", "Goals", "")]


@lru_cache(maxsize=8)
def _timeline(seed: int, removed: tuple):
    flags = tuple(f for f in ALL_FLAGS if f not in removed)
    m = MatchData(generate_match(seed, flags))
    return build_timeline(m), m.score_at


def counterfactual_for(moment: dict, actual_timeline, seed: int):
    key = moment.get("subtype") if moment["type"] == "TACTICAL_TURNING_POINT" else moment["type"]
    if key not in SCENARIOS:
        return None
    removed, question = SCENARIOS[key]
    cf_tl, cf_score = _timeline(seed, removed)
    m0 = int(moment["timestamp"] // 60)
    m1 = min(int(moment.get("end_timestamp", moment["timestamp"] + 600) // 60), m0 + 15, 90)
    team = moment["team"]
    if moment["type"] == "RED_CARD_IMPACT":
        team = "A" if team == "B" else "B"  # measure the beneficiary
        m0 += 1
    act, cf = window_metrics(actual_timeline, team, m0, m1), window_metrics(cf_tl, team, m0, m1)
    return {
        "kind": "synthetic_counterfactual",
        "label": "SYNTHETIC COUNTERFACTUAL: simulated, not observed",
        "question": question,
        "team": team,
        "window_minutes": [m0, m1],
        "removed_decisions": list(removed),
        "rows": [{"metric": label, "key": k, "unit": u, "actual": act.get(k), "counterfactual": cf.get(k)} for k, label, u in ROWS],
        "method": "Controlled re-simulation: same seed and scenario with the decision removed; identical analytics pipeline.",
    }
