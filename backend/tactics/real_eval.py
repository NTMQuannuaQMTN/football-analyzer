"""Tactical detectors on REAL matches (DFL), kept separate from the synthetic benchmark.

Labels used (no tactical labels exist for these matches, so each is stated for what it is):
  COUNTERATTACK  DFL's own annotation on every shot: ShotAtGoal CounterAttack="true|false" (human-coded by the
                 data provider). Task: is this shot the end of a counterattack? -> precision / recall / F1 over shots.
  FORMATION      official starting formation from DFL match information vs the first 15 minutes we detect
                 (exact and back-line agreement), plus how often the detected shape changes per half.
  HIGH_PRESS     WEAK, no label: association only. Share of opponent possessions that start in their own third and
                 are won back by the pressing team in the opponent half within 10 s, inside vs outside detected
                 press spells (lift). Event feed (DFL) is independent of the tracking-based press index.

    .venv/bin/python -m backend.tactics.real_eval            # writes data/eval/real/tactics_real.json
"""
import glob
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "eval" / "real" / "tactics_real.json"
LABELS = ROOT / "data" / "eval" / "real" / "dfl_shot_labels.json"
TEST_MATCHES = ("J03WMX", "J03WN1")  # held out while designing detectors; the 5 Düsseldorf matches are development


def shot_labels():
    """{file: [(t, team, counter_flag)]} for every DFL half file (cached)."""
    if LABELS.exists():
        return json.load(open(LABELS))
    from backend.reconstruction.dfl_match import SECTIONS, _ts, section_start
    out = {}
    for f in sorted(glob.glob(str(ROOT / "data" / "dfl_*_h[12].json"))):
        key = Path(f).stem
        mid = "DFL-MAT-" + key.split("_")[1]
        half = int(key[-1])
        meta = json.load(open(f))["meta"]
        t0 = section_start(mid, SECTIONS[half])
        info = glob.glob(str(ROOT / "data" / "dfl" / f"DFL_02_01_matchinformation_*_{mid}.xml"))[0]
        home = ET.parse(info).getroot().find(".//General").get("HomeTeamId")
        ev = glob.glob(str(ROOT / "data" / "dfl" / f"DFL_03_02_events_raw_*_{mid}.xml"))[0]
        rows = []
        for e in ET.parse(ev).getroot().iter("Event"):
            sh = e.find("ShotAtGoal") if e.find("ShotAtGoal") is not None else (e.find("Penalty/ShotAtGoal"))
            if sh is None or sh.get("CounterAttack") is None:
                continue
            t = _ts(e.get("CalculatedTimestamp") or e.get("EventTime")) - t0
            if -1 <= t <= meta["duration"] + 1:
                rows.append((round(t, 1), "A" if sh.get("Team") == home else "B", sh.get("CounterAttack") == "true"))
        out[key] = rows
    json.dump(out, open(LABELS, "w"))
    return out


def _prf(tp, fp, fn):
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / max(1e-9, p + r), 3),
            "tp": tp, "fp": fp, "fn": fn}


def counter_eval(intels, labels):
    """Pooled, and split into development / held-out test matches."""
    dev = {k: v for k, v in intels.items() if not any(t in k for t in TEST_MATCHES)}
    test = {k: v for k, v in intels.items() if any(t in k for t in TEST_MATCHES)}
    return {"all": _counter_eval(intels, labels), "dev (5 matches)": _counter_eval(dev, labels),
            "test (2 held-out matches)": _counter_eval(test, labels)}


def _counter_eval(intels, labels):
    tp = fp = fn = tn = 0
    per_match = {}
    for key, I in intels.items():
        shots = [e for e in I.m.events if e["event_type"] == "SHOT"]
        lab = labels.get(key, [])
        flagged = {c["shot"] for c in I.counters if c.get("shot")}
        for s in shots:
            gt = next((c for t, tm, c in lab if tm == s["team_id"] and abs(t - s["timestamp"]) <= 1.5), None)
            if gt is None:
                continue
            pred = s["id"] in flagged
            tp += pred and gt
            fp += pred and not gt
            fn += (not pred) and gt
            tn += (not pred) and (not gt)
        per_match[key] = len(I.counters)
    return {**_prf(tp, fp, fn), "true_negatives": tn, "shots_labelled": tp + fp + fn + tn,
            "detected_counters_per_half": per_match}


def formation_eval(intels):
    from backend.reconstruction.datasets import _match_info
    exact = back = n = 0
    changes = []
    rows = []
    for key, I in intels.items():
        mid = "DFL-MAT-" + key.split("_")[1]
        info = _match_info(mid)
        for tm in ("A", "B"):
            tl = I.timeline[tm]
            forms = [r.get("formation") for r in tl if r.get("formation") not in (None, "?")]
            changes.append(sum(1 for a, b in zip(forms, forms[1:]) if a != b))
            if key.endswith("h1"):
                first = [r.get("formation") for r in tl[:15] if r.get("formation") not in (None, "?")]
                det = max(set(first), key=first.count) if first else None
                off = (info.get("lineup_" + tm) or "").split()[0]
                n += 1
                exact += det == off
                back += bool(det) and det.split("-")[0] == off.split("-")[0]
                rows.append({"match": key, "team": tm, "official": info.get("lineup_" + tm), "detected": det})
    return {"teams": n, "exact_match": exact, "back_line_match": back,
            "detected_changes_per_team_half": round(float(np.mean(changes)), 2), "rows": rows}


def press_eval(intels):
    inside = [0, 0]
    outside = [0, 0]
    for key, I in intels.items():
        spans = [(p["team"], p["start"], p["end"]) for p in I.presses]
        for s in I.seqs:
            opp = s["team"]
            if not s["ball_own_x"] or s["ball_own_x"][0] > 35:
                continue  # opponent build-up from its own third
            presser = "B" if opp == "A" else "A"
            won = s["end"] - s["start"] <= 10 and (max(s["ball_own_x"][:11]) if s["ball_own_x"] else 0) < 52.5
            k = inside if any(tm == presser and a <= s["start"] < b for tm, a, b in spans) else outside
            k[0] += 1
            k[1] += won
    ri = inside[1] / max(1, inside[0])
    ro = outside[1] / max(1, outside[0])
    return {"label": "WEAK (association, no ground truth)", "buildups_inside_press": inside[0],
            "quick_regain_rate_inside": round(ri, 3), "buildups_outside_press": outside[0],
            "quick_regain_rate_outside": round(ro, 3), "lift": round(ri / ro, 2) if ro else None,
            "press_minutes_share": None}


def run():
    from backend.pipeline import MatchIntelligence
    labels = shot_labels()
    intels = {Path(f).stem: MatchIntelligence(7, f) for f in sorted(glob.glob(str(ROOT / "data" / "dfl_*_h[12].json")))}
    res = {"data": "DFL open data, 7 matches x 2 halves, ground-truth tracking (truth view)",
           "COUNTERATTACK (shots, DFL CounterAttack flag)": counter_eval(intels, labels),
           "FORMATION (official line-up, first 15 min)": formation_eval(intels),
           "HIGH_PRESS (weak association)": press_eval(intels)}
    total_press_min = sum(sum(p["end_min"] - p["start_min"] for p in I.presses) for I in intels.values())
    total_min = sum(2 * len(I.timeline["A"]) for I in intels.values())
    res["HIGH_PRESS (weak association)"]["press_minutes_share"] = round(total_press_min / max(1, total_min), 3)
    json.dump(res, open(OUT, "w"), indent=1)
    return res


if __name__ == "__main__":
    r = run()
    for k, v in r.items():
        if isinstance(v, dict):
            if k.startswith("COUNTER"):
                for part, x in v.items():
                    print(k, part, {a: b for a, b in x.items() if a not in ("detected_counters_per_half",)})
            else:
                print(k, {a: b for a, b in v.items() if a not in ("rows", "detected_counters_per_half")})
