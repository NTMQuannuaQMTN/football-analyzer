"""Tactical shift detection: change-points in a team's shape/pressing vector."""
from typing import Dict, List

import numpy as np

from backend.tactics.tactical_state import tactical_label

WIN = 5  # minutes either side
FEATURES = {"def_line": 4.0, "mid_line": 4.0, "width": 3.0, "depth": 3.5, "press_index": 15.0,
            "territory": 0.15}
THRESHOLD = 2.4


def _mean(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(v)) if v else 0.0


def detect_shifts(timeline: Dict[str, List[dict]]) -> List[dict]:
    out = []
    for team in ("A", "B"):
        rows = timeline[team]
        opp = timeline["B" if team == "A" else "A"]
        scores = []
        for k in range(len(rows)):
            if k < 3 or k > len(rows) - 3:
                scores.append(0.0)
                continue
            before, after = rows[max(0, k - WIN):k], rows[k:k + WIN]
            z = [(_mean(after, f) - _mean(before, f)) / s for f, s in FEATURES.items()]
            form_change = before[-1]["formation"] != after[min(len(after) - 1, 1)]["formation"]
            scores.append(float(np.sqrt(np.sum(np.square(z)))) + (1.5 if form_change else 0))
        # local maxima above threshold, 8+ minutes apart
        cands = sorted([k for k, s in enumerate(scores) if s >= THRESHOLD
                        and s == max(scores[max(0, k - 4):k + 5])], key=lambda k: -scores[k])
        chosen = []
        for k in cands:
            if all(abs(k - c) >= 8 for c in chosen):
                chosen.append(k)
        for k in sorted(chosen):
            before, after = rows[max(0, k - WIN):k], rows[k:k + WIN]
            deltas = {f: round(_mean(after, f) - _mean(before, f), 2) for f in FEATURES}
            f_after = after[min(len(after) - 1, 1)]["formation"]
            # first minute whose (per-minute) shape actually reads the new formation: a moment stamped earlier would
            # contradict the per-minute shape shown for that minute
            f_minute = next((k + i for i, r in enumerate(after) if r["formation"] == f_after), k)
            out.append({
                "type": "TACTICAL_SHIFT", "team": team, "minute": k, "start": k * 60,
                "score": round(scores[k], 2), "deltas": deltas,
                "formation_before": before[-1]["formation"], "formation_after": f_after, "formation_minute": f_minute,
                "label_before": tactical_label(before[-1], opp[k - 1]),
                "label_after": tactical_label(after[min(len(after) - 1, 2)], opp[min(len(opp) - 1, k + 2)]),
            })
    return out
