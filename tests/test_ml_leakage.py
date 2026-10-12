"""The live ML model must never see the future; replay is allowed to (retrospective)."""
import numpy as np
import pytest

from backend.reconstruction.ml import features


def _toy(T=120, N=6, seed=0):
    rng = np.random.default_rng(seed)
    pos = np.cumsum(rng.normal(0, 1, (T, N, 2)), 0) + [50, 34]
    vis = rng.random((T, N)) < 0.5
    on = np.ones((T, N), bool)
    team = np.array(["A"] * 3 + ["B"] * 3)
    role = np.array(["GK", "CB", "ST", "GK", "CB", "ST"])
    ball = np.cumsum(rng.normal(0, 2, (T, 2)), 0) + [50, 34]
    poss = rng.integers(0, 2, T)
    prior = pos + rng.normal(0, 3, pos.shape)
    return pos, vis, on, team, role, ball, poss, prior


@pytest.mark.parametrize("version", [1, 2, 3, 4])
def test_live_features_ignore_future(version):
    pos, vis, on, team, role, ball, poss, prior = _toy()
    T0 = 60
    vis2 = vis.copy()
    vis2[T0:] = ~vis2[T0:]
    pos2 = pos.copy()
    pos2[T0:] += 30  # completely different future
    ball2, poss2, prior2 = ball.copy(), poss.copy(), prior.copy()
    ball2[T0:] += 40
    poss2[T0:] = 1 - poss2[T0:]
    prior2[T0:] -= 20
    X1, i1 = features(np.where(vis[..., None], pos, np.nan), vis, on, team, role, ball, poss, None, prior, "live", t_index=range(T0), version=version)
    X2, i2 = features(np.where(vis2[..., None], pos2, np.nan), vis2, on, team, role, ball2, poss2, None, prior2, "live", t_index=range(T0), version=version)
    assert np.array_equal(i1, i2) and np.allclose(X1, X2)
    assert np.isfinite(X1).all()


@pytest.mark.parametrize("version", [1, 2, 3, 4])
def test_replay_features_do_use_future(version):
    pos, vis, on, team, role, ball, poss, prior = _toy()
    T0 = 60
    vis2 = vis.copy()
    vis2[T0:] = ~vis2[T0:]
    X1, _ = features(np.where(vis[..., None], pos, np.nan), vis, on, team, role, ball, poss, None, prior, "replay", t_index=range(T0), version=version)
    X2, _ = features(np.where(vis2[..., None], pos, np.nan), vis2, on, team, role, ball, poss, None, prior, "replay", t_index=range(T0), version=version)
    assert not np.allclose(X1, X2)
