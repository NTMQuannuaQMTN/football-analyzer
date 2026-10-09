"""Match Story graph: moments as nodes, temporal/causal-candidate links as edges, tactical chapters as bands.

Edges state what the data supports (sequence, co-occurrence), not proven causation."""
from typing import Dict, List

from backend.models.match import other
from backend.tactics.tactical_state import tactical_label

TACTIC_TYPES = ("TACTICAL_TURNING_POINT", "FORMATION_SHIFT")
OUTCOME_TYPES = ("GOAL_BUILD_UP", "COUNTERATTACK", "PRESSURE_BREAK")


def link_moments(moments: List[dict]) -> List[dict]:
    edges = []

    def add(a, b, relation, why):
        edges.append({"from": a["id"], "to": b["id"], "relation": relation, "explanation": why})
        a["links"].append({"to": b["id"], "relation": relation})

    for a in moments:
        w0, w1 = a.get("window", [a["timestamp"], a["timestamp"]])
        for b in moments:
            if a is b or b["timestamp"] < a["timestamp"]:
                continue
            in_win = w0 <= b["timestamp"] <= w1
            if a["type"] == "SUBSTITUTION_IMPACT" and b["type"] == "FORMATION_SHIFT" and b["team"] == a["team"] \
                    and b["timestamp"] - a["timestamp"] <= 480:
                add(a, b, "preceded", "The structural change followed the substitution")
            elif a["type"] == "SUBSTITUTION_IMPACT" and b["type"] == "GOAL_BUILD_UP" and a.get("player_id") in b["replay"]["focus_players"] and in_win:
                add(a, b, "involved_in", "The substitute was directly involved")
            elif a["type"] in TACTIC_TYPES and b["type"] in OUTCOME_TYPES and in_win:
                if b["team"] == a["team"]:
                    add(a, b, "followed_during", "Happened while the tactical change was active (temporal association)")
                elif b["type"] == "PRESSURE_BREAK":
                    add(a, b, "bypassed", "The press was bypassed while it was active")
            elif a["type"] == "RED_CARD_IMPACT" and b["type"] in OUTCOME_TYPES and b["team"] == other(a["team"]) and in_win:
                add(a, b, "followed", "Happened after the red card, with an extra player")
            elif a["type"] == "RED_CARD_IMPACT" and b["type"] in TACTIC_TYPES and b["team"] == other(a["team"]) \
                    and b["timestamp"] - a["timestamp"] <= 300:
                add(a, b, "followed", "Opponent changed shape after the red card")
    return edges


def chapters(timeline: Dict[str, List[dict]]) -> List[dict]:
    """Contiguous non-balanced tactical states per team (min 3 minutes)."""
    out = []
    for team in ("A", "B"):
        rows, opp = timeline[team], timeline[other(team)]
        labels = [tactical_label(r, o) for r, o in zip(rows, opp)]
        k = 0
        while k < len(labels):
            j = k
            while j < len(labels) and labels[j] == labels[k]:
                j += 1
            if labels[k] != "BALANCED" and j - k >= 4:
                out.append({"team": team, "label": labels[k], "start": k * 60, "end": j * 60})
            k = j
    return sorted(out, key=lambda c: c["start"])


def build_story(moments: List[dict], timeline) -> dict:
    edges = link_moments(moments)
    headline = [m for m in sorted(moments, key=lambda m: -m["importance"])][:8]
    return {
        "nodes": [{"id": m["id"], "type": m["type"], "team": m["team"], "timestamp": m["timestamp"],
                   "minute_label": m["minute_label"], "title": m["title"], "importance": m["importance"]}
                  for m in moments],
        "edges": edges,
        "chapters": chapters(timeline),
        "key_moment_ids": [m["id"] for m in sorted(headline, key=lambda m: m["timestamp"])],
    }
