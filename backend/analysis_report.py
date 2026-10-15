"""Grounded per-match analysis reports from the existing Match Insights Engine (no second analytics system).

For every catalogued match part it records what the engine found and tags every claim with its basis:
  ground_truth      positions from in-stadium tracking (DFL)                      -> facts
  recorded_event    official (DFL) or provider (SkillCorner CV) event feed         -> facts / provider facts
  observed          broadcast detections (SkillCorner) or our CV projections        -> measurements (unverified)
  model_inferred    hidden-player estimates                                        -> estimates
  heuristic         tactical labels / detectors / narrative interpretation         -> interpretation
Detectors without real-match validation are marked experimental (see docs/PROJECT_STATUS.md §5.4).

DFL halves are additionally analysed on the camera reconstruction (analytics re-run on `alt_frames.replay`) and the
tactical reading is scored against the ground-truth reading: formation per minute, defensive line / width /
compactness per minute and moment agreement. That measures how much the analysis depends on the reconstruction.

    .venv/bin/python -m backend.analysis_report --matches all
"""
import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "data" / "eval" / "match_reports"
OUT_MD = ROOT / "docs" / "match_reports"
VALIDATION = {  # real-data checks: DFL line-ups / CounterAttack flags; SkillCorner phase labels (backend/tactics/skillcorner_validation.py)
    "GOAL_BUILD_UP": "recorded goal; build-up sequence from events",
    "SUBSTITUTION_IMPACT": "recorded substitution; after-effects heuristic",
    "RED_CARD_IMPACT": "recorded card; after-effects heuristic",
    "FORMATION_SHIFT": "heuristic (template-fit formations: back line 13/14, exact 5/14 vs DFL line-ups; 57% minute agreement truth vs reconstruction)",
    "TACTICAL_TURNING_POINT": "heuristic; high press vs SkillCorner high-block minutes F1 0.05, low block F1 0.26 (experimental)",
    "PRESSURE_BREAK": "heuristic (experimental on real football)",
    "COUNTERATTACK": "heuristic; vs DFL CounterAttack flags F1 0.55; vs SkillCorner quick-break phases P 0.62 / R 0.23 (experimental)",
}

EVENT_LABELS = {"Build-up", "Shot", "Outcome", "Forced backwards", "Recovery", "Possession", "Change", "Card", "Goal evidence"}


def basis_of(I):
    src = I.m.meta.get("source") or {}
    if I.view == "truth":
        return {"positions": "ground_truth", "events": "recorded_event (official DFL feed)",
                "summary": "In-stadium tracking of all players (ground truth); official event feed"}
    if src.get("provider") == "SkillCorner" and src.get("event_feed"):
        return {"positions": "observed (SkillCorner broadcast detections) + model_inferred (hidden players)",
                "events": "recorded_event (SkillCorner CV-derived; not an official feed)",
                "summary": "Broadcast detections for visible players; every other position is a model estimate"}
    if src.get("provider") == "SkillCorner":
        return {"positions": "observed (SkillCorner legacy broadcast detections) + model_inferred", "events": "none",
                "summary": "Broadcast detections (legacy release, no event feed); hidden players estimated"}
    return {"positions": "observed (our CV from broadcast video; identity role-based, unverified) + model_inferred",
            "events": "labelled SoccerNet actions where available",
            "summary": "Computer vision from broadcast video; calibration and identity unverified (experimental)"}


def _minute_series(I, key):
    return {tm: np.array([r.get(key) if r.get(key) is not None else np.nan for r in I.timeline[tm]], float) for tm in ("A", "B")}


def minute_reliability(row):
    """Same formula as pipeline._attach_coverage: observed share + inferred share x mean inferred confidence
    (no calibrated confidence on video-derived files -> inferred players count as 0)."""
    obs, inf = row.get("observed_avg") or 0.0, row.get("inferred_avg") or 0.0
    share = obs / max(1e-6, obs + inf)
    conf = row.get("inferred_conf")
    return share + (1 - share) * (conf if conf is not None else 0.0)


def unreliable_segments(I, thr=0.5):
    """Runs of team-minutes whose visibility reliability < thr, or where the engine abstains (too few observed)."""
    out = []
    for tm in ("A", "B"):
        cur = None
        for k, row in enumerate(I.coverage[tm]):
            rel = minute_reliability(row)
            bad = rel < thr or bool(row.get("abstain"))
            if bad and cur is None:
                cur = {"team": tm, "from_minute": k, "to_minute": k, "min_reliability": round(rel, 2), "abstain_minutes": int(bool(row.get("abstain")))}
            elif bad:
                cur["to_minute"] = k
                cur["min_reliability"] = min(cur["min_reliability"], round(rel, 2))
                cur["abstain_minutes"] += int(bool(row.get("abstain")))
            elif cur is not None:
                out.append(cur)
                cur = None
        if cur is not None:
            out.append(cur)
    return out


def moment_record(mo, positions_basis):
    ev = []
    pos_tag = "ground_truth positions" if positions_basis == "ground_truth" else "observed + model_inferred positions"
    for e in mo.get("evidence", []):
        # the engine's own basis string says "observed events" for most items; tag by what the item is computed from
        if e["label"] in EVENT_LABELS:
            tag = "recorded_event"
        elif e["label"] in ("Tactical trigger", "Tactical context", "Transition"):
            tag = f"heuristic over {pos_tag}"
        else:
            tag = pos_tag
        ev.append({"label": e["label"], "detail": e["detail"], "value": e["value"], "unit": e["unit"], "t": e["timestamp"],
                   "engine_basis": e.get("basis"), "basis_tag": tag})
    cov = mo.get("coverage") or {}
    return {"id": mo["id"], "minute": mo.get("minute_label"), "t": mo["timestamp"], "type": mo["type"], "subtype": mo.get("subtype"),
            "team": mo["team"], "title": mo["title"], "importance": mo["importance"], "confidence": mo["confidence"],
            "reliability": cov.get("reliability"), "validation": VALIDATION.get(mo["type"], "heuristic"),
            "chain": mo.get("chain"), "consequences": mo.get("consequences"), "evidence": ev,
            "links": mo.get("links"), "hidden_state": mo.get("hidden_state")}


def analyse_part(I, until=None):
    s = I.summary(until)
    b = basis_of(I)
    moments = [moment_record(mo, b["positions"]) for mo in I.moments]
    return {"basis": b, "view": I.view, "duration_s": I.m.duration, "score": s.get("score"), "teams": s["teams"],
            "goals": s.get("goals"), "substitutions": s.get("substitutions"), "chapters": s.get("chapters"),
            "presses": [{"team": p["team"], "start": p["start"], "end": p["end"], "peak_press_index": p.get("peak_press_index")} for p in I.presses],
            "counterattacks": len(I.counters), "shifts": [{"team": x["team"], "minute": x["minute"], "from": x.get("label_before"), "to": x.get("label_after")} for x in I.shifts],
            "moments": moments, "story_links": sum(len(m.get("links") or []) for m in I.moments),
            "unreliable_segments": unreliable_segments(I), "observed_share": (I.m.meta.get("source") or {}).get("observed_share"),
            "changes": {tm: [{k: c.get(k) for k in ("minute_label", "type", "from", "to", "lasts_min", "moved", "deltas", "followed",
                                                     "preceded_by", "observed_share", "basis", "reading")} for c in getattr(I, "changes", {}).get(tm, [])]
                        for tm in ("A", "B")}}


def compare_truth_vs_recon(I_truth, I_rec):
    """How much of the tactical reading survives when analytics run on the camera reconstruction instead of truth."""
    out = {}
    for key in ("def_line", "width", "compactness", "press_index", "depth"):
        a, b = _minute_series(I_truth, key), _minute_series(I_rec, key)
        d = np.concatenate([np.abs(a[t] - b[t]) for t in ("A", "B")])
        d = d[np.isfinite(d)]
        out[f"{key}_minute_mae"] = round(float(d.mean()), 2) if len(d) else None
    same, tot = 0, 0
    for tm in ("A", "B"):
        for ra, rb in zip(I_truth.timeline[tm], I_rec.timeline[tm]):
            if ra.get("formation") and rb.get("formation"):
                tot += 1
                same += ra["formation"] == rb["formation"]
    out["formation_minute_agreement"] = round(same / max(1, tot), 3)
    pos_types = {"TACTICAL_TURNING_POINT", "FORMATION_SHIFT", "PRESSURE_BREAK", "COUNTERATTACK"}
    tm_ = [m for m in I_truth.moments if m["type"] in pos_types]
    rm_ = [m for m in I_rec.moments if m["type"] in pos_types]
    hit = sum(any(r["type"] == m["type"] and r["team"] == m["team"] and abs(r["timestamp"] - m["timestamp"]) <= 120 for r in rm_) for m in tm_)
    fa = sum(not any(m["type"] == r["type"] and m["team"] == r["team"] and abs(r["timestamp"] - m["timestamp"]) <= 120 for m in tm_) for r in rm_)
    out.update({"position_moments_truth": len(tm_), "position_moments_recon": len(rm_), "recalled_within_120s": hit,
                "recon_only": fa})
    return out


def recon_intel(path, mode="replay"):
    from backend.pipeline import MatchIntelligence
    raw = json.loads(Path(path).read_text())
    raw["frames"] = raw["alt_frames"][mode]
    raw.pop("alt_frames", None)
    raw.pop("reveals", None)
    raw["meta"] = dict(raw["meta"], view="vod")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(raw, fh)
        name = fh.name
    try:
        return MatchIntelligence(source=name)
    finally:
        os.unlink(name)


def report_match(entry):
    from backend.pipeline import MatchIntelligence
    rec = {"id": entry["id"], "kind": entry["kind"], "label": entry["label"], "info": entry.get("info"), "parts": {}}
    for part, path in entry["parts"].items():
        if path is None:
            continue
        I = MatchIntelligence(source=path)
        r = analyse_part(I)
        r["file"] = str(Path(path).relative_to(ROOT))
        src = I.m.meta.get("source") or {}
        r["reconstruction_model"] = src.get("reconstruction_model") or (I.m.meta.get("reconstruction") or {}).get("model")
        r["goals_not_in_provider_events"] = src.get("goals_not_in_provider_events")
        r["official_final_score"] = I.m.meta.get("final_score") if src.get("goals_not_in_provider_events") else None
        if I.view == "truth" and "replay" in I.alt_frames:
            Ir = recon_intel(path)
            r["reconstruction_view"] = analyse_part(Ir)
            r["reconstruction_view"]["basis"] = {"positions": "observed (exact tracking, simulated camera) + model_inferred",
                                                 "events": "recorded_event", "summary": "Analytics re-run on the camera reconstruction"}
            r["truth_vs_reconstruction"] = compare_truth_vs_recon(I, Ir)
        rec["parts"][part] = r
    return rec


def markdown(rec):
    L = [f"# {rec['label']}: {rec['id']}", ""]
    for part, r in rec["parts"].items():
        ts = r["teams"]
        names = {}
        L += [f"## {part}", "", f"**Data basis:** {r['basis']['summary']}. Positions: `{r['basis']['positions']}`. Events: `{r['basis']['events']}`.", ""]
        if r.get("reconstruction_model"):
            L += [f"Hidden-player model: `{r['reconstruction_model']}`.", ""]
        if r.get("goals_not_in_provider_events"):
            off = r["official_final_score"]
            L += [f"**Score caveat:** official final score {off['A']}–{off['B']}; {sum(r['goals_not_in_provider_events'].values())} goal(s) "
                  "do not appear in the provider event stream, so their time and scorer are unknown and they are absent from the timeline.", ""]
        sc = r.get("score") or {}
        L += ["| | A | B |", "|---|---:|---:|"]
        for k, lab in (("goals", "Goals"), ("shots", "Shots"), ("xg", "xG (engine model)"), ("passes", "Passes"),
                       ("pass_accuracy", "Pass accuracy"), ("possession_seconds", "Possession (s)"), ("press_index", "Press index (heuristic)"),
                       ("formation", "Last formation (heuristic)")):
            L.append(f"| {lab} | {ts['A'].get(k)} | {ts['B'].get(k)} |")
        L += ["", "**Formations over time (heuristic):** " + "; ".join(
            f"{tm}: " + ", ".join(f"{f['from_minute']}' {f['formation']}" for f in ts[tm].get("formations") or []) for tm in ("A", "B")), ""]
        if r["presses"]:
            L.append("**High-press spells (experimental detector):** " + ", ".join(f"{p['team']} {int(p['start'] // 60)}'–{int(p['end'] // 60)}'" for p in r["presses"]))
        L += [f"**Counterattacks detected (experimental):** {r['counterattacks']}", ""]
        L += ["### Timeline", "", "| Min | Type | Team | Title | Confidence | Visibility reliability | Validation |", "|---|---|---|---|---:|---:|---|"]
        for m in r["moments"]:
            L.append(f"| {m['minute']} | {m['type']} | {m['team']} | {m['title']} | {m['confidence']} | {m['reliability'] if m['reliability'] is not None else 'n/a'} | {m['validation']} |")
        top = sorted(r["moments"], key=lambda m: -m["importance"])[:3]
        if top:
            L += ["", "### Evidence chains (top 3 by importance)", ""]
        for m in top:
            L += [f"**{m['minute']} {m['title']}** ({m['type']}). Chain: " + " → ".join(f"{c['stage']}: {c['label']}" for c in (m["chain"] or [])), ""]
            for e in m["evidence"]:
                L.append(f"- [{e['basis_tag']}] {e['label']}: {e['detail']}")
            if m.get("consequences"):
                L.append(f"- Followed by (not proof of causation): {', '.join(m['consequences'])}")
            if m.get("hidden_state"):
                L.append(f"- Camera / reconstruction / truth at this moment: `{json.dumps(m['hidden_state'])[:300]}`")
            L.append("")
        if any(r.get("changes", {}).values()):
            L += ["### Tactical & formation changes (every change, both teams)", ""]
            for tm in ("A", "B"):
                L.append(f"**Team {tm}** ({len(r['changes'][tm])} changes)")
                for c in r["changes"][tm]:
                    L.append(f"- {c['minute_label']} [{c['type']}] {c['reading']} _({c['basis']})_")
                L.append("")
        if r.get("truth_vs_reconstruction"):
            c = r["truth_vs_reconstruction"]
            L += ["### Does the analysis survive reconstruction? (truth vs camera reconstruction)", "",
                  f"Formation agreement per team-minute: {c['formation_minute_agreement']:.0%}. Mean absolute difference per minute: "
                  f"defensive line {c['def_line_minute_mae']} m, width {c['width_minute_mae']} m, compactness {c['compactness_minute_mae']} m, "
                  f"press index {c['press_index_minute_mae']}. Position-based moments: truth {c['position_moments_truth']}, "
                  f"reconstruction {c['position_moments_recon']}, recalled within ±120 s {c['recalled_within_120s']}, reconstruction-only {c['recon_only']}.", ""]
        if r["unreliable_segments"]:
            L += ["### Unreliable segments (visibility reliability < 0.5, or engine abstained)", ""]
            for u in r["unreliable_segments"]:
                L.append(f"- Team {u['team']}, minutes {u['from_minute']}–{u['to_minute']} (min reliability {u['min_reliability']}; "
                         f"abstained {u['abstain_minutes']} min)")
            L.append("")
        L += ["### Limitations", "",
              "- Tactical labels, formations, press and counter detections are heuristics. Moments are linked by order in time, not by proven cause.",
              "- " + ("Positions are ground truth; the camera view is simulated." if r["view"] == "truth" else
                      "Hidden players are model estimates with a calibrated 80% radius; observed positions are provider or CV measurements, not verified truth."),
              ""]
    return "\n".join(L)


def main(ids=None):
    from backend.catalog import _catalog_static
    OUT_JSON.mkdir(parents=True, exist_ok=True)
    OUT_MD.mkdir(parents=True, exist_ok=True)
    index = []
    for e in _catalog_static():
        if e["kind"] == "synthetic" or (ids and e["id"] not in ids):
            continue
        try:
            rec = report_match(e)
        except Exception as err:  # record and continue
            index.append({"id": e["id"], "status": "failed", "error": repr(err)[:300]})
            print("FAILED", e["id"], repr(err)[:200], flush=True)
            continue
        (OUT_JSON / f"{e['id']}.json").write_text(json.dumps(rec, indent=1, default=float))
        (OUT_MD / f"{e['id']}.md").write_text(markdown(rec))
        index.append({"id": e["id"], "status": "ok", "label": e["label"], "parts": list(rec["parts"])})
        print("ok", e["id"], flush=True)
    (OUT_JSON / "index.json").write_text(json.dumps(index, indent=1))
    return index


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", default="all")
    a = ap.parse_args()
    main(None if a.matches == "all" else a.matches.split(","))
