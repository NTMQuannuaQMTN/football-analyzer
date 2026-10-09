"""OBSERVE: what a broadcast camera would show of a synthetic match (deterministic, seeded).

The simulator knows every player's position (ground truth). Here we hide players exactly the way a
single moving broadcast camera does, so reconstruction can be scored against the truth.

Modes
  broadcast  camera follows the ball with lag, zooms, cuts to close-ups (nobody visible) ~15% of the time
  k=<n>      the n on-pitch players nearest the camera centre are visible (controlled visibility levels)
The ball is treated as observed (the camera follows it).
"""
from dataclasses import dataclass

import numpy as np

from backend.models.match import L, W, MatchData


@dataclass
class Observation:
    mode: str
    vis: np.ndarray        # (T, N) bool — player visible this second
    cam_x: np.ndarray      # (T,) camera centre (for display / debugging)
    cut: np.ndarray        # (T,) bool — camera on a close-up/replay: nobody visible

    def coverage(self, on_pitch: np.ndarray) -> float:
        return float(self.vis.sum() / max(1, on_pitch.sum()))


def _camera_centre(ball_x: np.ndarray) -> np.ndarray:
    cam = np.empty_like(ball_x)
    c = ball_x[0]
    for t, bx in enumerate(ball_x):
        c += 0.5 * (bx - c)
        cam[t] = np.clip(c, 22, L - 22)
    return cam


def observe(m: MatchData, mode: str = "broadcast", seed: int = 7) -> Observation:
    rng = np.random.default_rng(seed)
    T, N = m.xy.shape[:2]
    on = ~np.isnan(m.xy[..., 0])
    cam = _camera_centre(m.ball[:, 0])
    vis = np.zeros((T, N), dtype=bool)
    cut = np.zeros(T, dtype=bool)
    if mode.startswith("k="):
        k = int(mode[2:])
        for t in range(T):
            d = np.hypot(m.xy[t, :, 0] - cam[t], (m.xy[t, :, 1] - m.ball[t, 1]) * 0.5)
            d[~on[t]] = np.inf
            vis[t, np.argsort(d)[:k]] = True
        vis &= on
        return Observation(mode, vis, cam, cut)
    # broadcast camera
    zoom, zoom_left, cut_left = 1.0, 0, 0
    for t in range(T):
        if zoom_left <= 0:
            zoom, zoom_left = rng.uniform(0.85, 1.25), int(rng.integers(8, 25))
        zoom_left -= 1
        if cut_left > 0:
            cut_left -= 1
            cut[t] = True
            continue
        if rng.random() < 0.025:
            cut_left = int(rng.integers(3, 10))
            cut[t] = True
            continue
        half_w = (17 + 0.22 * m.xy[t, :, 1]) * zoom      # wider on the far side (perspective)
        vis[t] = on[t] & (np.abs(m.xy[t, :, 0] - cam[t]) < half_w) & (rng.random(N) > 0.06)  # detector misses
    return Observation(mode, vis, cam, cut)
