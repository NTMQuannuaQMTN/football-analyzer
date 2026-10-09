"""Attacking / defensive transitions: how the team *shape* responds to a turnover.

A ball recovery (observable event) becomes an
  ATTACKING_TRANSITION for the winner if its outfield centroid advances >= ADV m within WINDOW s,
  DEFENSIVE_TRANSITION for the loser if its outfield centroid retreats >= ADV m within WINDOW s.
Positions may be observed or reconstructed — this is what makes transition detection sensitive to
hidden-player estimation.
"""
import numpy as np

from backend.models.match import MatchData

ADV, WINDOW = 4.0, 8


def _centroid_x(m: MatchData, team: str) -> np.ndarray:
    mask = m.team_mask(team)
    x = m.own_xy(team)[:, mask, 0]
    with np.errstate(all="ignore"), np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        return np.nanmean(x, axis=1)


def detect_transitions(m: MatchData):
    cx = {t: _centroid_x(m, t) for t in ("A", "B")}
    att, dfn = [], []
    for e in m.events:
        if e["event_type"] != "POSSESSION_CHANGE" or e.get("cause") not in ("tackle", "interception"):
            continue
        t0 = int(e["timestamp"])
        if t0 + WINDOW >= m.T:
            continue
        win, lose = e["team_id"], ("B" if e["team_id"] == "A" else "A")
        a = cx[win][t0 + 2:t0 + WINDOW + 1] - cx[win][t0]
        d = cx[lose][t0 + 2:t0 + WINDOW + 1] - cx[lose][t0]
        if np.isfinite(a).any() and np.nanmax(a) >= ADV:
            att.append({"team": win, "t": float(e["timestamp"])})
        if np.isfinite(d).any() and np.nanmin(d) <= -ADV:
            dfn.append({"team": lose, "t": float(e["timestamp"])})
    return att, dfn
