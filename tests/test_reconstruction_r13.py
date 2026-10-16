"""Regression tests for the 13 October reconstruction work: causality of the temporal model, held-out guards,
match-level splits, real-broadcast re-entry extraction, SkillCorner orientation, catalog grouping, measured
positions preserved in re-inferred exports, and the goal-without-shot engine path."""
import json
from pathlib import Path

import numpy as np
import pytest

from backend.reconstruction import hidden_models as hm

ROOT = Path(__file__).resolve().parents[1]


def _toy(T=60, N=6, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform([0, 0], [105, 68], (T, N, 2))
    vis = rng.random((T, N)) < 0.5
    on = np.ones((T, N), bool)
    team = np.array(["A", "A", "A", "B", "B", "B"])[:N]
    role = np.array(["GK", "CB", "ST", "GK", "CM", "W"])[:N]
    ball = rng.uniform([0, 0], [105, 68], (T, 2))
    poss = rng.integers(-1, 2, T)
    alive = rng.random(T) < 0.8
    prior = rng.uniform([0, 0], [105, 68], (T, N, 2))
    return pos, vis, on, team, role, ball, poss, alive, prior


def _future_changed(arrs, t0, seed=1):
    pos, vis, on, team, role, ball, poss, alive, prior = [a.copy() for a in arrs]
    rng = np.random.default_rng(seed)
    pos[t0 + 1:] = rng.uniform([0, 0], [105, 68], pos[t0 + 1:].shape)
    vis[t0 + 1:] = rng.random(vis[t0 + 1:].shape) < 0.5
    ball[t0 + 1:] = rng.uniform([0, 0], [105, 68], ball[t0 + 1:].shape)
    poss[t0 + 1:] = rng.integers(-1, 2, len(poss) - t0 - 1)
    alive[t0 + 1:] = ~alive[t0 + 1:]
    prior[t0 + 1:] = rng.uniform([0, 0], [105, 68], prior[t0 + 1:].shape)
    return pos, vis, on, team, role, ball, poss, alive, prior


def _windows(arrs, mode, t, j):
    pos, vis, on, team, role, ball, poss, alive, prior = arrs
    C = hm.context_tensor(np.where(vis[..., None], pos, np.nan), vis, on, team, ball, poss, alive, prior)
    return hm.CtxStore([C], mode).windows(np.zeros(len(t), np.int64), t, j)


def test_live_context_window_never_reads_the_future():
    a = _toy()
    t0 = 30
    b = _future_changed(a, t0)
    t = np.arange(0, t0 + 1)
    j = np.zeros_like(t)
    assert np.array_equal(_windows(a, "live", t, j), _windows(b, "live", t, j))


def test_replay_context_window_is_retrospective():
    a = _toy()
    b = _future_changed(a, 30)
    t, j = np.array([28]), np.array([1])
    assert not np.array_equal(_windows(a, "replay", t, j), _windows(b, "replay", t, j))


def test_live_gru_prediction_is_causal_end_to_end():
    import torch
    a = _toy()
    t0 = 30
    b = _future_changed(a, t0)
    from backend.reconstruction.ml import features
    pos, vis, on, team, role, ball, poss, alive, prior = a
    X, _ = features(np.where(vis[..., None], pos, np.nan), vis, on, team, role, ball, poss, alive, prior, "live", version=3)
    torch.manual_seed(0)
    net = hm.make_gru(X.shape[1], "live")
    ck = {"arch": "gru_ctx", "mode": "live", "state": net.state_dict(), "n_in": X.shape[1], "feat_version": 3,
          "calibration": [{"lo": -np.inf, "hi": np.inf, "r80": 5.0, "p5": 0.5}]}
    model = hm.HiddenModel(ck)
    out = []
    for arrs in (a, b):
        pos, vis, on, team, role, ball, poss, alive, prior = arrs
        idx, corr, sig = hm.predict_segment(model, pos, vis, on, team, role, ball, poss, alive, prior, "live",
                                            t_index=range(0, t0 + 1))
        out.append((idx, corr, sig))
    assert np.array_equal(out[0][0], out[1][0])
    assert np.allclose(out[0][1], out[1][1]) and np.allclose(out[0][2], out[1][2])


def test_benchmark_refuses_checkpoints_that_saw_the_test_match():
    from backend.reconstruction.benchmark import check_held_out
    check_held_out({"train": ["M1", "M2"], "test": "M3"}, "M3")
    with pytest.raises(RuntimeError):
        check_held_out({"train": ["M1", "M3"], "test": "M3"}, "M3")
    with pytest.raises(RuntimeError):
        check_held_out({"train": ["M1", "M2"], "test": None}, "M3")  # a 'final' model has no held-out match


def test_fold_split_is_match_level():
    from backend.reconstruction.train_hidden import fold_split
    ms = [f"M{i}" for i in range(7)]
    for i in range(7):
        train, test = fold_split(i, ms)
        assert test == ms[i] and test not in train and len(train) == 6
    assert fold_split("final", ms) == (ms, None)


def test_reentries_need_a_prior_sighting_and_a_gap():
    from backend.reconstruction.gsr_benchmark import reentries
    vis = np.array([[0, 1], [1, 1], [0, 0], [0, 1], [1, 1]], bool)
    # player 0: first seen at t=1 (not a re-entry), hidden 2-3, back at 4 (gap 2). player 1: hidden at 2, back at 3.
    assert sorted(reentries(vis)) == [(3, 1, 1), (4, 0, 2)]


def test_skillcorner_event_coordinates_undo_attacking_side_normalisation():
    from backend.reconstruction.skillcorner_match import event_xy
    assert event_xy("10", "5", "left_to_right", False, 105, 68) == (62.5, 39.0)
    assert event_xy("10", "5", "right_to_left", False, 105, 68) == (42.5, 29.0)
    assert event_xy("10", "5", "right_to_left", True, 105, 68) == (62.5, 39.0)
    assert event_xy("", "5", "left_to_right", False, 105, 68) == (None, None)


def test_catalog_groups_skillcorner_halves(tmp_path, monkeypatch):
    from backend import catalog
    meta = {"duration": 10, "teams": {"A": {"name": "Home FC", "color": "#000"}, "B": {"name": "Away FC", "color": "#fff"}},
            "player_order": [], "initial_score": {"A": 0, "B": 0},
            "source": {"type": "computer_vision", "provider": "SkillCorner", "competition": "A-League 2024/25",
                       "date": "2025-01-01", "observed_share": 0.6}}
    for h in (1, 2):
        (tmp_path / f"skc_123_h{h}.json").write_text(json.dumps({"meta": meta, "players": [], "events": [], "frames": []}))
    monkeypatch.setattr(catalog, "DATA", tmp_path)
    catalog._catalog_static.cache_clear()
    try:
        entries = [e for e in catalog._catalog_static() if e["id"] == "skc-123"]
        assert len(entries) == 1 and list(entries[0]["parts"]) == ["h1", "h2"]
        card = next(c for c in catalog.lobby() if c["id"] == "skc-123")
        assert "model estimate" in card["note"] and card["kind"] == "computer_vision"
    finally:
        catalog._catalog_static.cache_clear()


@pytest.mark.skipif(not (ROOT / "cv" / "weights" / "hidden_live_final_v3.pt").exists(), reason="weights not present")
def test_reinferred_export_keeps_measured_positions_and_never_reads_stored_estimates():
    from backend.reconstruction.apply_exports import clip_from_export
    from backend.reconstruction.gsr_benchmark import model_path
    from backend.reconstruction.skillcorner_match import reconstruct
    T, N = 40, 4
    rng = np.random.default_rng(3)
    order = ["A1", "A2", "B1", "B2"]
    players = [{"id": o, "team": o[0], "position": "GK" if o.endswith("1") else "MID", "role_slot": "GK" if o.endswith("1") else "CM"}
               for o in order]
    frames = []
    for t in range(T):
        obs = [bool(rng.random() < 0.5) for _ in order]
        xy = [[round(float(rng.uniform(0, 105)), 1), round(float(rng.uniform(0, 68)), 1)] for _ in order]
        frames.append({"t": t, "b": [52.5, 34.0], "p": "A", "xy": xy, "obs": obs})
    d = {"meta": {"player_order": order}, "players": players, "frames": frames}
    clip = clip_from_export(d)
    assert np.isnan(clip["pos"][~np.array([f["obs"] for f in frames])]).all()  # stored estimates discarded
    models = {m: hm.load_hidden(model_path("final_v3", m)) for m in ("live", "replay")}
    rec = reconstruct(clip, models)
    for mode in ("live", "replay"):
        pos, r80, p5, vis = rec[mode]
        assert np.isfinite(pos[clip["on"] & ~vis]).all()
        assert np.allclose(clip["pos"][vis], np.array([[f["xy"][j] for j in range(N)] for f in frames])[vis])


def test_goal_without_a_located_shot_does_not_break_the_engine(tmp_path):
    from backend.pipeline import MatchIntelligence
    from simulator.match_generator import generate_match
    raw = generate_match(7)
    raw.pop("truth", None)
    goals = [e for e in raw["events"] if e["event_type"] == "GOAL"]
    assert goals
    raw["events"] = [e for e in raw["events"] if e["event_type"] != "SHOT"]
    raw["meta"]["view"] = "truth"
    p = tmp_path / "m.json"
    p.write_text(json.dumps(raw))
    I = MatchIntelligence(source=str(p))
    assert sum(m["type"] == "GOAL_BUILD_UP" for m in I.moments) == len(goals)


@pytest.mark.skipif(not (ROOT / "data" / "skillcorner" / "data" / "matches" / "1886347").exists(), reason="SkillCorner data not present")
def test_skillcorner_players_stay_on_pitch_through_provider_tracking_gaps():
    from backend.reconstruction.skillcorner_benchmark import load_match
    name, clip = next(load_match(str(ROOT / "data" / "skillcorner" / "data" / "matches" / "1886347")))
    gaps = ~clip["tracked"]
    assert gaps.any()  # replays / close-ups: the provider lists nobody
    assert (clip["on"][gaps].sum(1) >= 20).all()  # but the match sheet keeps ~22 players on the pitch
    assert not np.isfinite(clip["pos"][gaps]).any()  # and nothing is "observed" there
    assert clip["ball_seen"].mean() < 1.0  # ball detection flag recorded before forward fill


@pytest.mark.skipif(not (ROOT / "cv" / "weights" / "hidden_live_final_v3.pt").exists(), reason="weights not present")
def test_estimates_stay_on_the_pitch_next_to_off_pitch_measurements():
    from backend.reconstruction.gsr_benchmark import model_path
    from backend.reconstruction.skillcorner_match import reconstruct
    T, N = 30, 4
    pos = np.full((T, N, 2), np.nan)
    pos[:, 0] = [-6.0, 30.0]  # a measured player standing well outside the touchline
    pos[::3, 1] = [5.0, 2.0]
    clip = {"pos": pos, "on": np.ones((T, N), bool), "ball": np.tile([20.0, 30.0], (T, 1)),
            "team": np.array(["A", "A", "B", "B"]), "role": np.array(["CB", "FB", "GK", "ST"]),
            "poss": np.zeros(T, int), "alive": np.ones(T, bool), "anchors": None}
    models = {m: hm.load_hidden(model_path("final_v3", m)) for m in ("live", "replay")}
    for mode, (p, _, _, vis) in reconstruct(clip, models).items():
        est = clip["on"] & ~vis
        assert (p[est][:, 0] >= 0).all() and (p[est][:, 0] <= 105).all()
        assert (p[est][:, 1] >= 0).all() and (p[est][:, 1] <= 68).all()
        assert np.allclose(p[vis], pos[vis])  # measurements untouched, even off the pitch


def test_goal_without_a_known_scorer_serialises_through_the_api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from backend import catalog
    from backend.api.main import app
    from simulator.match_generator import generate_match
    raw = generate_match(7)
    raw.pop("truth", None)
    for e in raw["events"]:
        if e["event_type"] == "GOAL":
            e["player_id"], e["assist_id"] = None, None
    raw["events"] = [e for e in raw["events"] if e["event_type"] != "SHOT"]
    raw["meta"]["view"] = "truth"
    (tmp_path / "dfl_TEST_h1.json").write_text(json.dumps(raw))
    monkeypatch.setattr(catalog, "DATA", tmp_path)
    catalog._catalog_static.cache_clear()
    try:
        r = TestClient(app).get("/match/moments", params={"match": "dfl-TEST", "part": "h1"})
        assert r.status_code == 200
        assert all(None not in m["replay"]["focus_players"] for m in r.json())
    finally:
        catalog._catalog_static.cache_clear()


def test_goal_is_credited_to_the_shot_whose_ball_enters_the_goal_not_a_blocked_one():
    """skc-1886347 2nd half: Mata's shot at 2599 s bounced back out; De Vries' rebound at 2601.1 s went in (2603 s)."""
    from backend.reconstruction.skillcorner_match import attribute_goals
    T = 40
    ball = np.tile([60.0, 34.0], (T, 1))
    ball[10:13] = [[93.7, 32.4], [93.3, 35.6], [88.1, 36.5]]  # first shot comes back out
    ball[13:16] = [[99.8, 34.5], [106.1, 34.4], [107.0, 35.0]]  # rebound goes in at t=14 and stays in the net
    ball[16:30] = [107.0, 35.0]
    ball[30:] = [52.5, 34.0]                                     # kick-off follows
    seen = np.ones(T, bool)
    first = {"timestamp": 10.0, "event_type": "SHOT", "team_id": "A", "player_id": "A9", "x": 94.7, "y": 35.8, "xg": 0.3,
             "provider_lead_to_goal": True}
    rebound = {"timestamp": 12.1, "event_type": "SHOT", "team_id": "A", "player_id": "A15", "x": 87.1, "y": 35.7, "xg": 0.2}
    ev = [first, rebound]
    attribute_goals(ev, [("A", 10.0, {"A": 1, "B": 0})], ball, seen)
    goal = next(e for e in ev if e["event_type"] == "GOAL")
    assert goal["player_id"] == "A15" and goal["timestamp"] == 14.0 and goal["evidence"] == "ball tracked entering the goal"
    assert first.get("outcome") != "goal" and rebound["outcome"] == "goal"


def test_goal_without_credible_evidence_is_not_pinned_on_a_shot():
    """No provider goal-ending possession and no ball seen in the net: the goal counts in the official score but no
    GOAL event / moment is invented on a nearby shot (that produced false goal moments)."""
    from backend.reconstruction.skillcorner_match import attribute_goals
    T = 30
    ball = np.tile([60.0, 34.0], (T, 1))
    seen = np.zeros(T, bool)  # camera on a replay: ball never tracked
    shot = {"timestamp": 5.0, "event_type": "SHOT", "team_id": "B", "player_id": "B7", "x": 10.0, "y": 34.0, "xg": 0.4}
    ev = [shot]
    unlocated = attribute_goals(ev, [("B", 20.0, {"A": 0, "B": 1})], ball, seen)
    assert not [e for e in ev if e["event_type"] == "GOAL"] and shot.get("outcome") != "goal"
    assert unlocated == [{"team": "B", "score_after": {"A": 0, "B": 1}, "provider_score_change_s": 20.0}]


def test_rebound_crossing_under_the_bar_is_a_goal_even_if_tracking_drifts_behind_the_goal():
    """skc-1886347 (a real goal, confirmed by the user): De Vries' shot crossed the line at 1.65 m; tracking then drifts
    behind the goal. With the score change and a kick-off, that is a goal."""
    from backend.reconstruction.skillcorner_match import attribute_goals
    T = 120
    ball = np.tile([60.0, 34.0], (T, 1))
    ball[10:16] = [[99.8, 34.5], [106.1, 34.4], [106.8, 36.1], [107.6, 36.5], [110.8, 37.5], [112.2, 39.4]]
    z = np.full(T, 0.3)
    z[11] = 1.65
    ball[95:] = [52.5, 34.0]
    shot = {"timestamp": 9.0, "event_type": "SHOT", "team_id": "A", "player_id": "A15", "x": 87.0, "y": 35.0, "xg": 0.1}
    ev = [shot]
    unlocated = attribute_goals(ev, [("A", 8.0, {"A": 2, "B": 0})], ball, np.ones(T, bool), ball_z=z)
    goal = [e for e in ev if e["event_type"] == "GOAL"]
    assert not unlocated and goal and goal[0]["player_id"] == "A15" and goal[0]["timestamp"] == 11.0


def test_stray_goal_restart_label_is_not_drawn_as_a_goal():
    """skc-1886347 41:42: a goal kick, but the next provider row is labelled 'restart after a goal'; no goal counted."""
    from backend.reconstruction.skillcorner_match import display_ball
    T = 200
    clip = {"ball": np.tile([30.0, 34.0], (T, 1)), "ball_seen": np.ones(T, bool),
            "stoppages": [(20.0, 110.0, "Goal kick", "goal_kick_against")]}
    _, labels, _, _ = display_ball(clip, [])
    assert "Goal · kick-off" not in labels and labels[50] == "Goal kick"
    clip["stoppages"] = [(20.0, 110.0, "Goal · kick-off", "goal_for")]  # provider goal label, no counted goal
    _, labels, _, _ = display_ball(clip, [])
    assert "Goal · kick-off" not in labels and labels[50] == "Dead ball"


def test_set_piece_goal_splits_the_stoppage_and_is_not_timed_at_the_kick_off():
    from backend.reconstruction.skillcorner_match import attribute_goals, display_ball
    T = 200
    ball = np.tile([90.0, 10.0], (T, 1))
    ball[101] = [105.6, 33.0]  # the corner, taken ~60 s after it was awarded, is headed in at t=101
    seen = np.ones(T, bool)
    ev = []
    # provider zero-length goal row at t=100; previous possession ended at t=40 (corner awarded)
    attribute_goals(ev, [("A", 100.0, {"A": 1, "B": 0})], ball, seen, goal_ends=[(100.0, "A", "A9", False, 41.0)])
    goal = [e for e in ev if e["event_type"] == "GOAL"][0]
    assert goal["timestamp"] == 101.0 and goal["player_id"] == "A9"  # at the goal row, not at the corner award
    clip = {"ball": ball, "ball_seen": seen, "stoppages": [(41.0, 99.0, "Corner", "corner_for"), (102.0, 160.0, "Goal · kick-off", "goal_for")]}
    _, labels, _, _ = display_ball(clip, [goal])
    assert labels[50] == "Corner" and labels[110] == "Goal · kick-off"


def test_ball_over_the_crossbar_is_not_a_goal():
    from backend.reconstruction.skillcorner_match import ball_entry
    ball = np.tile([106.0, 34.0], (10, 1))
    assert ball_entry(ball, np.ones(10, bool), "A", 0, 5, ball_z=np.full(10, 3.5)) is None
    assert ball_entry(ball, np.ones(10, bool), "A", 0, 5, ball_z=np.full(10, 1.0)) == 0


def test_formation_shift_moment_is_stamped_where_the_new_shape_is_first_read():
    from backend.tactics.shift import detect_shifts
    def row(m, f, line):
        return {"minute": m, "formation": f, "def_line": line, "mid_line": line + 10, "width": 40, "depth": 30,
                "press_index": 30, "territory": 0.5, "possession": 0.5, "def_line_oop": line, "pressures_pm": 5,
                "final_third_recoveries": 0, "high_press": False, "def_line_oop_s": line, "territory_s": 0.5}
    a = [row(m, "4-2-3-1", 30) for m in range(20)] + [row(20, "4-2-3-1", 45)] + [row(m, "4-4-2", 45) for m in range(21, 40)]
    shifts = [s for s in detect_shifts({"A": a, "B": [row(m, "4-4-2", 35) for m in range(40)]}) if s["team"] == "A"]
    assert shifts
    for s in shifts:
        assert a[s["formation_minute"]]["formation"] == s["formation_after"]


def test_live_summary_only_shows_moments_whose_evidence_is_already_in_the_past():
    from backend.pipeline import MatchIntelligence
    I = MatchIntelligence(source=str(ROOT / "data" / "dfl_J03WR9_r13_h1.json")) if (ROOT / "data" / "dfl_J03WR9_r13_h1.json").exists() else MatchIntelligence(7)
    assert all(m["available_at"] >= m["timestamp"] for m in I.moments)
    late = [m for m in I.moments if m["available_at"] > m["timestamp"] + 60]
    assert late  # some moments need later evidence (e.g. a shift confirmed by the following minutes)
    m = late[0]
    T = (m["timestamp"] + m["available_at"]) / 2  # after the moment started, before its evidence is complete
    ids = {h["id"] for h in I.summary(until=T)["highlights"]}
    assert m["id"] not in ids
    assert m["id"] in {h["id"] for h in I.summary(until=m["available_at"])["highlights"]}


def test_lobby_hides_superseded_versions_but_keeps_them_reachable(tmp_path, monkeypatch):
    from backend import catalog
    meta = lambda src: {"duration": 10, "teams": {"A": {"name": "H", "color": "#000"}, "B": {"name": "A", "color": "#fff"}},
                        "player_order": [], "initial_score": {"A": 0, "B": 0}, "source": src}
    files = {"dfl_X1_h1.json": {"type": "tracking"}, "dfl_X1_r13_h1.json": {"type": "tracking", "reconstruction_session": "r13"},
             "dfl_X1_trial7h_preview_h1.json": {"type": "tracking", "session": "seven-hour", "experimental": True},
             "cv_clip_a.json": {"type": "computer_vision"}, "cv_r13_clip_a.json": {"type": "computer_vision"}}
    for name, src in files.items():
        (tmp_path / name).write_text(json.dumps({"meta": meta(src), "players": [], "events": [], "frames": []}))
    monkeypatch.setattr(catalog, "DATA", tmp_path)
    monkeypatch.setattr(catalog, "LIVE_MATCH", "none")
    monkeypatch.setattr(catalog, "SHOW_SUPERSEDED", False)
    catalog._catalog_static.cache_clear()
    try:
        shown = {c["id"] for c in catalog.lobby()}
        assert {"dfl-X1-r13", "cv-r13-clip-a"} <= shown
        assert not shown & {"dfl-X1", "dfl-X1-trial7h-preview", "cv-clip-a"}
        assert catalog.entry("dfl-X1") is not None  # still reachable by URL
        monkeypatch.setattr(catalog, "LIVE_MATCH", "dfl-X1")
        assert not catalog.superseded(catalog.entry("dfl-X1"))  # the live match is never hidden
    finally:
        catalog._catalog_static.cache_clear()


def test_possession_changes_from_provider_possessions():
    from backend.reconstruction.skillcorner_match import possession_changes
    def pp(fs, fe, pid, x, y, side="left_to_right"):
        return {"event_type": "player_possession", "player_id": str(pid), "frame_start": str(fs), "frame_end": str(fe),
                "x_start": str(x), "y_start": str(y), "attacking_side": side}
    rows = [pp(0, 20, 1, 0, 0), pp(30, 60, 2, 5, 0),            # A -> A: no change
            pp(70, 90, 11, -10, 3),                               # A -> B with a regain by B: tackle
            {"event_type": "on_ball_engagement", "player_id": "11", "frame_start": "62", "frame_end": "72", "end_type": "direct_regain"},
            pp(2000, 2020, 1, 0, 0),                               # B -> A after a long stoppage on the centre spot: kickoff
            pp(2025, 2040, 12, 20, 10)]                            # A -> B quickly: loose ball
    pid_of = {1: "A1", 2: "A2", 11: "B11", 12: "B12"}
    team_of = {1: "A", 2: "A", 11: "B", 12: "B"}
    xy = lambda x, y, r: (52.5 + float(x), 34 + float(y))
    ev = possession_changes(rows, pid_of, team_of, xy, lambda f: int(f) / 10, 400)
    pc = [e for e in ev if e["event_type"] == "POSSESSION_CHANGE"]
    assert [e["cause"] for e in pc] == ["tackle", "kickoff", "loose_ball"]
    assert [e["team_id"] for e in pc] == ["B", "A", "B"]
    assert sum(e["event_type"] == "KICK_OFF" for e in ev) == 2  # period start + centre-spot restart


def test_untracked_ball_is_flagged_for_display():
    from backend.reconstruction.skillcorner_match import frames_for
    T, N = 3, 1
    clip = {"on": np.ones((T, N), bool), "pos": np.full((T, N, 2), np.nan), "ball": np.tile([50.0, 30.0], (T, 1))}
    rec = (np.full((T, N, 2), 10.0), np.full((T, N), 5.0), np.full((T, N), 0.5), np.zeros((T, N), bool))
    fr = frames_for(clip, rec, ["A"] * T, np.array([True, False, True]))
    assert [f["bs"] for f in fr] == ["tracked", "held_or_unknown", "tracked"]


def test_ball_stays_in_the_net_after_a_goal_until_the_kick_off():
    from backend.reconstruction.skillcorner_match import display_ball
    T = 120
    ball = np.tile([60.0, 34.0], (T, 1))
    seen = np.ones(T, bool)
    ball[20] = [106.0, 34.5]            # goal at t=20 (home attacks +x)
    ball[21:60] = [[70 + k % 7, 20 + k % 11] for k in range(39)]  # celebrations: ball wanders / replays
    seen[25:55] = False
    ball[90:] = [52.5, 34.0]           # kick-off from the centre spot at t=90
    clip = {"ball": ball, "ball_seen": seen, "stoppages": []}
    b, labels, _, basis = display_ball(clip, [{"timestamp": 20.0, "team_id": "A"}])
    assert all(labels[t] == "Goal · kick-off" for t in range(22, 90))
    assert all(b[t, 0] > 105 and abs(b[t, 1] - 34) <= 3.2 for t in range(22, 86))  # in the net, not wandering
    assert np.allclose(b[95], [52.5, 34.0]) and labels[95] is None


def test_throw_in_ball_is_placed_on_the_touchline():
    from backend.reconstruction.skillcorner_match import display_ball
    T = 40
    ball = np.tile([50.0, 30.0], (T, 1))
    ball[10:13] = [[50, 66], [50, 69], [51, 70]]  # goes out over the far touchline
    clip = {"ball": ball, "ball_seen": np.ones(T, bool), "stoppages": [(12.0, 25.0, "Throw-in", "throw_in_for")]}
    b, labels, _, _ = display_ball(clip, [])
    assert labels[11] == "Throw-in"  # spell starts when the ball left the pitch, not a second later
    assert all(abs(b[t, 1] - 68.5) < 1e-6 for t in range(14, 23))  # placed on the touchline until thrown


def test_restart_is_taken_from_where_the_tracked_ball_rests_and_never_teleports():
    """The user's goal kick: the ball was put on a rule spot, then jumped to the receiver in the other half."""
    from backend.reconstruction.skillcorner_match import display_ball, glide_gaps
    T = 80
    ball = np.tile([40.0, 30.0], (T, 1))
    seen = np.ones(T, bool)
    for k in range(3, 10):                                    # play moves towards the goal (physical speeds)
        ball[k] = [40.0 - 5.2 * (k - 2), 30.0 + 1.4 * (k - 2)]
    ball[10:13] = [[4.0, 40.0], [-1.0, 42.0], [-2.0, 43.0]]  # goes out over the goal line
    ball[13:30] = [3.0, 38.0]                                 # tracked at rest in the goal area: the kick spot
    for k in range(30, 34):                                   # goal kick: flies ~60 m to the other half
        ball[k] = [3.0 + 15 * (k - 29), 38.0 - (k - 29)]
    ball[34:] = ball[33]
    clip = {"ball": ball, "ball_seen": seen, "stoppages": [(12.0, 33.0, "Goal kick", "goal_kick_for")]}
    b, labels, _, _ = display_ball(clip, [])
    b = glide_gaps(b, seen, labels)
    assert np.allclose(b[20], [3.0, 38.0])                     # placed where the tracking shows it, not a rule spot
    assert labels[29] == "Goal kick" and labels[30] is None    # play restarts at the kick, not at the reception
    steps = [float(np.hypot(*(b[t] - b[t - 1]))) for t in range(1, T)]
    jumps = [t for t, d in zip(range(1, T), steps) if d > 32 and not (labels[t] and not labels[t - 1])]
    assert not jumps                                            # only allowed jump: at the end of play


def test_goal_stoppage_never_swallows_the_stoppages_after_it():
    """skc-2015213: a goal whose kick-off label is missing must not stretch over later throw-ins / free kicks."""
    from backend.reconstruction.skillcorner_match import display_ball
    T = 400
    clip = {"ball": np.tile([30.0, 34.0], (T, 1)), "ball_seen": np.ones(T, bool),
            "stoppages": [(10.0, 300.0, "Goal · kick-off", "goal_for"), (100.0, 120.0, "Throw-in", "throw_in_for"),
                          (200.0, 220.0, "Free kick", "free_kick_against")]}
    _, labels, _, _ = display_ball(clip, [{"timestamp": 9.0, "team_id": "A"}])
    assert labels[50] == "Goal · kick-off" and labels[110] == "Throw-in" and labels[210] == "Free kick"
    assert labels[150] is None  # live play between the stoppages


def test_tactical_changes_cover_every_change_with_moved_players_and_ignore_blips():
    from backend.tactics.changes import _runs, tactical_changes
    assert [r[0] for r in _runs(list("BBBDBBBLLLBB"), 2)] == ["B", "L", "B"]  # the 1-minute "D" is noise
    from backend.pipeline import MatchIntelligence
    I = MatchIntelligence(source=str(ROOT / "data" / "skc_2017461_h1.json")) if (ROOT / "data" / "skc_2017461_h1.json").exists() else MatchIntelligence(7)
    ch = tactical_changes(I)
    for tm in ("A", "B"):
        forms = [r.get("formation") for r in I.timeline[tm]]
        n_form = sum(1 for k in range(1, len(forms)) if forms[k] and forms[k - 1] and forms[k] != forms[k - 1])
        assert sum(c["type"] == "FORMATION" for c in ch[tm]) == n_form  # every formation change, none invented
        for c in ch[tm]:
            assert c["reading"] and c["timestamp"] == c["minute"] * 60
            if c["type"] == "FORMATION":
                before = {p for line in I.timeline[tm][c["minute"] - 1]["formation_lines"] for p in line}
                assert {x["player_id"] for x in c["moved"]} <= before
    first = min((c for tm in "AB" for c in ch[tm]), key=lambda c: c["available_at"])
    live = tactical_changes(I, until=first["available_at"] - 1)  # live: just before its after-window has elapsed
    assert first["id"] not in {c["id"] for tm in "AB" for c in live[tm]}
    assert all(c["available_at"] <= first["available_at"] - 1 for tm in "AB" for c in live[tm])


def _zoo_checkpoint(n_in, test="M3"):
    import torch
    torch.manual_seed(0)
    return {"arch": "zoo_top3", "n_in": n_in, "mode": "live", "fold": 0, "train": ["M1", "M2"], "test": test,
            "feat_version": 3, "calibration": None,
            "members": {k: hm.make_zoo_net(k, n_in).state_dict() for k in ("mlp", "mdn", "zone")}}


def test_top3_ensemble_predicts_a_finite_correction_and_spread_for_every_row():
    X = np.random.default_rng(0).normal(size=(50, 66)).astype(np.float32)
    mu, sig = hm.HiddenModel(_zoo_checkpoint(66)).raw(X)
    assert mu.shape == (50, 2) and sig.shape == (50,)
    assert np.isfinite(mu).all() and (sig > 0).all()
    # the zone classifier's expected cell centre can never leave its grid
    assert np.abs(hm.zoo_centres()).max() < hm.ZOO_ZONE_R


def test_blend_of_ensembles_loads_members_relative_to_itself_and_keeps_the_held_out_match(tmp_path):
    import torch
    from backend.reconstruction.benchmark import check_held_out
    for tag in ("a", "b"):
        (tmp_path / tag).mkdir()
        torch.save(_zoo_checkpoint(66), tmp_path / tag / "hidden_live_fold0.pt")
    (tmp_path / "blend").mkdir()
    torch.save({"arch": "ensemble", "members": ["../a/hidden_live_fold0.pt", "../b/hidden_live_fold0.pt"], "mode": "live",
                "fold": 0, "train": ["M1", "M2"], "test": "M3", "feat_version": 3, "calibration": None},
               tmp_path / "blend" / "hidden_live_fold0.pt")
    m = hm.load_hidden(tmp_path / "blend" / "hidden_live_fold0.pt")
    assert len(m.members) == 2 and all(x.arch == "zoo_top3" for x in m.members)
    check_held_out(m.ck, "M3")
    mu, sig = m.raw(np.zeros((4, 66), np.float32))
    assert mu.shape == (4, 2) and np.isfinite(sig).all()
