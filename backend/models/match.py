"""MatchData: numpy view over a raw synthetic (or, later, real/streamed) match.

INGEST layer. A future Azure Event Hubs consumer only needs to produce the same raw dict
({meta, players, events, frames}) to feed the rest of the pipeline.
"""
from typing import Dict, List

import numpy as np

L, W = 105.0, 68.0
TEAMS = ("A", "B")


def other(team: str) -> str:
    return "B" if team == "A" else "A"


def to_own(team: str, x, y):
    return (x, y) if team == "A" else (L - x, W - y)


class MatchData:
    def __init__(self, raw: dict):
        self.raw = raw
        self.meta = raw["meta"]
        self.players: List[dict] = raw["players"]
        self.player_by_id: Dict[str, dict] = {p["id"]: p for p in self.players}
        self.events: List[dict] = raw["events"]
        self.frames: List[dict] = raw["frames"]
        self.order: List[str] = self.meta["player_order"]
        self.idx = {pid: i for i, pid in enumerate(self.order)}
        self.T = len(self.frames)
        self.duration = self.meta["duration"]
        xy = np.full((self.T, len(self.order), 2), np.nan)
        for i, f in enumerate(self.frames):
            for j, p in enumerate(f["xy"]):
                if p is not None:
                    xy[i, j] = p
        self.xy = xy
        self.ball = np.array([f["b"] for f in self.frames], dtype=float)
        self.poss = np.array([{"A": 0, "B": 1}.get(f["p"], -1) for f in self.frames])
        self.team_of = np.array([0 if pid.startswith("A") else 1 for pid in self.order])
        self.is_gk = np.array([self.player_by_id[pid].get("position") == "GK" for pid in self.order])

    def team_mask(self, team: str, outfield: bool = True) -> np.ndarray:
        m = self.team_of == (0 if team == "A" else 1)
        if outfield:
            m &= ~self.is_gk
        return m

    def own_xy(self, team: str) -> np.ndarray:
        """All player coordinates in `team`'s attacking frame (T, N, 2)."""
        if team == "A":
            return self.xy
        out = self.xy.copy()
        out[..., 0] = L - out[..., 0]
        out[..., 1] = W - out[..., 1]
        return out

    def own_ball(self, team: str) -> np.ndarray:
        if team == "A":
            return self.ball
        return np.stack([L - self.ball[:, 0], W - self.ball[:, 1]], axis=1)

    def events_between(self, start: float, end: float, types=None, team=None) -> List[dict]:
        return [e for e in self.events if start <= e["timestamp"] < end
                and (types is None or e["event_type"] in types)
                and (team is None or e["team_id"] == team)]

    def score_at(self, t: float) -> Dict[str, int]:
        s = dict(self.meta.get("initial_score") or {"A": 0, "B": 0})
        for e in self.events:
            if e["event_type"] == "GOAL" and e["timestamp"] <= t:
                s[e["team_id"]] += 1
        return s
