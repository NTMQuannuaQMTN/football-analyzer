"""Product guarantees: inferred vs observed, uncertainty, schemas, story links, counterfactual labels, LLM guard."""
import glob
import json

import numpy as np
import pytest

from backend.pipeline import MatchIntelligence, get_intel

DFL = sorted(glob.glob("data/dfl_*_h1.json"))


def test_inferred_positions_are_marked_and_carry_uncertainty():
    I = get_intel()
    fr = I.frames("replay")
    inf = [(f, j) for f in fr[::30] for j, o in enumerate(f["obs"]) if o is False]
    assert inf, "the camera view must hide some players"
    for f, j in inf:
        assert f["conf"][j] is not None and 0 <= f["conf"][j] <= 1
        assert f["rad"][j] is not None and f["rad"][j] > 0
    obs = [(f, j) for f in fr[::30] for j, o in enumerate(f["obs"]) if o is True]
    assert all(f["rad"][j] == 0 or f["rad"][j] is None for f, j in obs)


def test_reconstruction_state_schema():
    I = get_intel()
    for st in I.player_states(600, "replay"):
        assert set(st) >= {"player_id", "position", "state_source", "confidence", "uncertainty_radius", "evidence"}
        assert st["state_source"] in ("observed", "inferred")
        if st["state_source"] == "inferred":
            assert st["uncertainty_radius"] > 0 and st["confidence"] < 1


def test_tactical_outputs_valid():
    I = get_intel()
    for c in I.counters:
        assert c["team"] in "AB" and c["passes"] <= 4 and c["progression_m"] >= 25 and c["opponents_goal_side"] <= 7
    for p in I.presses:
        assert p["end_min"] - p["start_min"] >= 3
    for tm in "AB":
        for r in I.timeline[tm]:
            f = r["formation"]
            assert f == "?" or all(x.isdigit() for x in f.split("-"))


def test_story_links_valid():
    I = get_intel()
    ids = {m["id"] for m in I.moments}
    for e in I.story["edges"]:
        assert e["from"] in ids and e["to"] in ids
        assert e.get("relation", "") not in ("caused", "causes", "because")


def test_counterfactuals_are_labelled_synthetic():
    I = get_intel()
    cfs = [m["counterfactual"] for m in I.moments if m.get("counterfactual")]
    assert cfs
    assert all(c["kind"] == "synthetic_counterfactual" and "SYNTHETIC" in c["label"] for c in cfs)


@pytest.mark.skipif(not DFL, reason="DFL match files not built")
def test_real_match_hidden_state_and_reveals():
    raw = json.load(open(DFL[0]))
    for r in raw["reveals"][:200]:
        assert r["hidden_s"] >= 5 and r["radius_m"] > 0
        err = float(np.hypot(r["predicted"][0] - r["actual"][0], r["predicted"][1] - r["actual"][1]))
        assert abs(err - r["error_m"]) < 0.2  # reported error is the real distance
    I = MatchIntelligence(7, DFL[0])
    for m in I.moments:
        h = m.get("hidden_state")
        if h:
            assert h["observed_players"] + h["inferred_players"] <= h["outfield_players"] + 0.01
            assert h["metrics"]["truth"] and h["metrics"]["reconstructed"]


def test_llm_narrative_rejects_invented_numbers():
    from backend.narratives.generator import FoundryNarrativeGenerator
    I = get_intel()
    mo = next(m for m in I.moments if m["type"] == "GOAL_BUILD_UP")
    ctx = {"teams": I.m.meta["teams"]}
    g = FoundryNarrativeGenerator()
    g.endpoint, g.key = "http://test", "k"
    g._call = lambda payload: {"why_it_matters": "They had 87.3% possession and 14 shots.", "fan_text": "Wow."}
    assert g.generate(mo, ctx)["source"] == "template"
    g._call = lambda payload: {"why_it_matters": "A goal followed the spell of possession.", "fan_text": "Goal!"}
    out = g.generate(mo, ctx)
    assert out["source"] == "foundry" and out["why_it_matters"].startswith("A goal followed")
