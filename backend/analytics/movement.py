import numpy as np

from backend.models.match import MatchData

MAX_HUMAN_SPEED = 11.0  # m/s; larger jumps are restarts / tracking glitches


def speeds(m: MatchData) -> np.ndarray:
    """(T, N) speed in m/s (1 Hz frames). NaN when off pitch or implausible."""
    d = np.linalg.norm(np.diff(m.xy, axis=0), axis=2)
    d[d > MAX_HUMAN_SPEED] = np.nan
    return np.vstack([np.full((1, d.shape[1]), np.nan), d])


def player_movement(m: MatchData, spd: np.ndarray, t_end: int = None) -> dict:
    t_end = m.T if t_end is None else min(m.T, int(t_end) + 1)
    s = spd[:t_end]
    out = {}
    with np.errstate(all="ignore"):
        dist = np.nansum(s, axis=0)
        top = np.nanmax(np.where(np.isnan(s), -1, s), axis=0)
        sprints = np.nansum(s > 7.0, axis=0)
    for pid, j in m.idx.items():
        out[pid] = {"distance_km": round(float(dist[j]) / 1000, 2),
                    "top_speed_kmh": round(max(0.0, float(top[j])) * 3.6, 1),
                    "sprint_seconds": int(sprints[j])}
    return out
