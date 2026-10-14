"""Validate the engine's tactical detectors on real broadcast matches against SkillCorner's phase-of-play labels.

SkillCorner Open Data labels every phase of play for both teams (in possession: build_up / create / finish / direct /
quick_break / transition / chaotic / set_play; out of possession: high_block / medium_block / low_block / ...).
These are PROVIDER labels computed from provider tracking (proxy labels, not human ground truth).

  counterattack  engine COUNTERATTACK vs provider quick_break / transition phases of the same team (+-10 s)
  high press     engine per-minute HIGH_PRESS flag vs provider minutes where the defending team is mostly in high_block
  low block      engine LOW_BLOCK label vs provider minutes mostly in low_block

    .venv/bin/python -m backend.tactics.skillcorner_validation
"""
import csv
import glob
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SC = ROOT / "data" / "skillcorner" / "data" / "matches"
COUNTER_TYPES = {"quick_break", "transition"}
TOL_S = 10


def provider_phases(mid, period, f0):
    meta = json.load(open(glob.glob(str(SC / mid / "*_match.json"))[0]))
    home = meta["home_team"]["id"]
    rows = list(csv.DictReader(open(glob.glob(str(SC / mid / "*_phases_of_play.csv"))[0])))
    out = []
    for r in rows:
        if r["period"] != str(period):
            continue
        tm = "A" if int(r["team_in_possession_id"]) == home else "B"
        out.append({"team": tm, "start": (int(r["frame_start"]) - f0) / 10, "end": (int(r["frame_end"]) - f0) / 10,
                    "in": r["team_in_possession_phase_type"], "out": r["team_out_of_possession_phase_type"]})
    return out


def _f0(mid, period):
    meta = json.load(open(glob.glob(str(SC / mid / "*_match.json"))[0]))
    f0 = int(next(p for p in meta["match_periods"] if p["period"] == period)["start_frame"])
    return f0 + (-f0) % 10  # first sampled frame, as in skillcorner_match.export


def minute_block(phases, team, n_min):
    """Per minute: share of the team's DEFENDING time in high_block / low_block (None if it barely defended)."""
    hi, lo, tot = np.zeros(n_min), np.zeros(n_min), np.zeros(n_min)
    for p in phases:
        if p["team"] == team:
            continue  # the other team has the ball: `team` is defending, with block type p["out"]
        for k in range(max(0, int(p["start"] // 60)), min(n_min, int(p["end"] // 60) + 1)):
            a, b = max(p["start"], k * 60), min(p["end"], (k + 1) * 60)
            if b > a:
                tot[k] += b - a
                hi[k] += (b - a) * (p["out"] == "high_block")
                lo[k] += (b - a) * (p["out"] == "low_block")
    ok = tot >= 15
    return np.where(ok, hi / np.maximum(tot, 1), np.nan), np.where(ok, lo / np.maximum(tot, 1), np.nan)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * p * r / (p + r) if p and r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": None if p is None else round(p, 3),
            "recall": None if r is None else round(r, 3), "f1": round(f, 3)}


def run():
    from backend.pipeline import MatchIntelligence
    c_tp = c_fp = c_fn = 0
    hp = {"tp": 0, "fp": 0, "fn": 0}
    lb = {"tp": 0, "fp": 0, "fn": 0}
    press_hi, press_lo, line_hi, line_lo = [], [], [], []
    per = []
    for f in sorted(glob.glob(str(ROOT / "data" / "skc_*_h*.json"))):
        mid, half = Path(f).stem.split("_")[1], int(Path(f).stem[-1])
        I = MatchIntelligence(source=f)
        ph = provider_phases(mid, half, _f0(mid, half))
        counters_p = [p for p in ph if p["in"] in COUNTER_TYPES]
        det = I.counters
        tp = sum(any(p["team"] == c["team"] and p["start"] - TOL_S <= c["start"] <= p["end"] + TOL_S for p in counters_p) for c in det)
        rec = sum(any(c["team"] == p["team"] and p["start"] - TOL_S <= c["start"] <= p["end"] + TOL_S for c in det) for p in counters_p)
        c_tp += tp
        c_fp += len(det) - tp
        c_fn += len(counters_p) - rec
        for tm in ("A", "B"):
            rows = I.timeline[tm]
            n = len(rows)
            hi, lo = minute_block(ph, tm, n)
            for k, r in enumerate(rows):
                if np.isnan(hi[k]):
                    continue
                prov_hi, prov_lo = hi[k] >= 0.5, lo[k] >= 0.5
                eng_hi = bool(r.get("high_press"))
                from backend.tactics.tactical_state import tactical_label
                eng_lo = tactical_label(r, I.timeline["B" if tm == "A" else "A"][k]) == "LOW_BLOCK"
                for d, e, pv in ((hp, eng_hi, prov_hi), (lb, eng_lo, prov_lo)):
                    d["tp"] += int(e and pv)
                    d["fp"] += int(e and not pv)
                    d["fn"] += int((not e) and pv)
                if r.get("press_index") is not None:
                    (press_hi if prov_hi else press_lo if prov_lo else []).append(r["press_index"])
                if r.get("def_line_oop") is not None:
                    (line_hi if prov_hi else line_lo if prov_lo else []).append(r["def_line_oop"])
        per.append({"file": Path(f).name, "provider_counter_phases": len(counters_p), "engine_counters": len(det), "matched": tp})
    res = {"label_source": "SkillCorner phases_of_play (provider labels from provider tracking; proxy, not human ground truth)",
           "counterattack": {**prf(c_tp, c_fp, c_fn), "note": "match = same team, engine start within the provider quick_break/transition phase +-10 s"},
           "high_press_minutes": {**prf(hp["tp"], hp["fp"], hp["fn"]), "note": "provider minute = defending team >= 50% of its defending time in high_block"},
           "low_block_minutes": {**prf(lb["tp"], lb["fp"], lb["fn"]), "note": "provider minute = >= 50% in low_block"},
           "press_index_mean": {"provider_high_block_minutes": round(float(np.mean(press_hi)), 1) if press_hi else None,
                                "provider_low_block_minutes": round(float(np.mean(press_lo)), 1) if press_lo else None,
                                "n": [len(press_hi), len(press_lo)]},
           "def_line_oop_mean_m": {"provider_high_block_minutes": round(float(np.mean(line_hi)), 1) if line_hi else None,
                                   "provider_low_block_minutes": round(float(np.mean(line_lo)), 1) if line_lo else None},
           "per_half": per}
    out = ROOT / "data" / "eval" / "real" / "tactics_skillcorner.json"
    out.write_text(json.dumps(res, indent=1, default=int))
    return res


if __name__ == "__main__":
    r = run()
    print(json.dumps({k: v for k, v in r.items() if k != "per_half"}, indent=1))
