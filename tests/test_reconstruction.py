"""Hidden-state reconstruction: accuracy vs synthetic ground truth, uncertainty, and no future leakage in live mode."""
import numpy as np

from backend.models.match import MatchData
from backend.reconstruction.estimator import SRC_INF, SRC_OBS, reconstruct, roster_timeline
from backend.reconstruction.evaluate import hold_last_baseline
from backend.reconstruction.observation import observe
from simulator.match_generator import generate_match

M = MatchData(generate_match(7))
ON, _ = roster_timeline(M)


def _hidden_mae(pos, vis):
    hid = ON & ~vis & ~np.isnan(M.xy[..., 0])
    return float(np.nanmean(np.linalg.norm(pos - M.xy, axis=2)[hid]))


def test_beats_hold_last_baseline_at_every_visibility():
    for lvl in ("k=14", "k=6", "broadcast"):
        vis = observe(M, lvl, 7).vis & ON
        base = _hidden_mae(hold_last_baseline(M, vis, ON), vis)
        for mode in ("live", "replay"):
            assert _hidden_mae(reconstruct(M, vis, mode).pos, vis) < base, (lvl, mode)


def test_observed_players_are_exact_and_labelled():
    vis = observe(M, "broadcast", 7).vis & ON
    rec = reconstruct(M, vis, "replay")
    assert np.all(rec.src[vis] == SRC_OBS) and np.all(rec.src[ON & ~vis] == SRC_INF)
    assert np.allclose(rec.pos[vis], M.xy[vis])
    inf = rec.src == SRC_INF
    assert np.all(rec.rad[inf] > 0) and np.all((rec.conf[inf] > 0) & (rec.conf[inf] < 1))


def test_live_mode_never_uses_the_future():
    """Changing observations after time T must not change live estimates before T."""
    vis = observe(M, "broadcast", 7).vis & ON
    T = 3000
    vis2 = vis.copy()
    vis2[T:] = ~vis2[T:] & ON[T:]  # completely different future
    a, b = reconstruct(M, vis, "live"), reconstruct(M, vis2, "live")
    assert np.allclose(a.pos[:T], b.pos[:T], equal_nan=True)
    r1, r2 = reconstruct(M, vis, "replay"), reconstruct(M, vis2, "replay")
    assert not np.allclose(r1.pos[:T], r2.pos[:T], equal_nan=True)  # replay *does* look ahead (retrospective)
