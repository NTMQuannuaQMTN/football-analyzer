"""Ground-truth regression: the engine must rediscover the scripted scenario."""
from backend.pipeline import get_intel
from simulator.match_generator import generate_match


def test_reproducible():
    a, b = generate_match(7), generate_match(7)
    assert a["frames"][3000] == b["frames"][3000] and len(a["events"]) == len(b["events"])


def test_detects_scripted_story():
    I = get_intel()
    assert I.m.meta["final_score"] == {"A": 3, "B": 1}
    types = {(m["type"], m["team"], int(m["timestamp"] // 60)) for m in I.moments}
    assert any(t == "TACTICAL_TURNING_POINT" and team == "A" and 14 <= mn <= 18 for t, team, mn in types)  # high press
    assert any(t == "TACTICAL_TURNING_POINT" and team == "B" and 27 <= mn <= 33 for t, team, mn in types)  # low block (scripted 30', +-3 min as in the benchmark)
    assert any(t == "FORMATION_SHIFT" and team == "A" and 63 <= mn <= 67 for t, team, mn in types)        # 4-2-3-1
    assert any(t == "RED_CARD_IMPACT" for t, _, _ in types)
    assert sum(t == "GOAL_BUILD_UP" for t, _, _ in types) == 4
    shift = next(m for m in I.moments if m["type"] == "FORMATION_SHIFT")
    assert shift["detection"]["formation_after"] == "4-2-3-1"
    assert any(e["relation"] == "preceded" for e in I.story["edges"])  # sub -> shift (temporal, not causal)


def test_evidence_is_from_analytics():
    for m in get_intel().moments:
        assert m["evidence"] and all(e["source"] in ("analytics", "reconstruction") for e in m["evidence"])
        assert all("basis" in e for e in m["evidence"])  # observed events vs observed + inferred positions


def _numbers(obj, out):
    import re
    if isinstance(obj, dict):
        for v in obj.values():
            _numbers(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _numbers(v, out)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        out.add(round(abs(float(obj)), 1))
    elif isinstance(obj, str):
        for n in re.findall(r"\d+(?:\.\d+)?", obj):
            out.add(round(float(n), 1))
    return out


def test_narratives_use_only_engine_numbers():
    """Every number in a narrative must come from the moment's structured data (no invented statistics)."""
    import re
    for m in get_intel().moments:
        facts = _numbers({k: v for k, v in m.items() if k != "narrative"}, set())
        n = m["narrative"]
        texts = [n["why_it_matters"], n["fan"]["text"], n["broadcast"]["lower_third"], n.get("visibility_note") or ""]
        for txt in texts:
            for num in re.findall(r"\d+(?:\.\d+)?", txt.replace("4-2-3-1", "").replace("4-3-3", "").replace("4-4-2", "").replace("4-4-1", "")):
                v = round(float(num), 1)
                pct = round(v / 100, 2)
                assert v in facts or round(v) in facts or pct in {round(f, 2) for f in facts} or any(abs(v - f) <= 0.6 for f in facts), (m["id"], txt, num)
