"""Per-frame team shape metrics and formation detection (all in the team's own frame)."""
import numpy as np

from backend.models.match import L, W

from backend.models.match import MatchData

NEAR_BALL_M = 12.0
MIN_VISIBLE = 7  # team-shape metrics need most of the team in view (broadcast video shows ~half the pitch)


def frame_shape(m: MatchData, team: str) -> dict:
    mask = m.team_mask(team)
    p = m.own_xy(team)[:, mask]               # (T, n, 2)
    x, y = p[..., 0], p[..., 1]
    bx = m.own_ball(team)
    with np.errstate(all="ignore"), np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        xs = np.sort(x, axis=1)                    # NaNs last
        def_line = np.nanmean(xs[:, :4], axis=1)
        mid_line = np.nanmedian(x, axis=1)
        width = np.nanmax(y, axis=1) - np.nanmin(y, axis=1)
        depth = np.nanmax(x, axis=1) - np.nanmin(x, axis=1)
        cx, cy = np.nanmean(x, axis=1), np.nanmean(y, axis=1)
        compact = np.nanmean(np.hypot(x - cx[:, None], y - cy[:, None]), axis=1)
        partial = np.sum(~np.isnan(x), axis=1) < MIN_VISIBLE
        for arr in (def_line, mid_line, width, depth, cx, compact):
            arr[partial] = np.nan
        dball = np.hypot(x - bx[:, :1], y - bx[:, 1:])
        near = np.sum(dball < NEAR_BALL_M, axis=1)
        # closing speed of the 3 nearest players: reduction of distance to ball per second
        prev = np.vstack([dball[:1], dball[:-1]])
        closing = prev - dball
        closing[np.abs(closing) > 11] = np.nan
        order = np.argsort(np.where(np.isnan(dball), 1e9, dball), axis=1)[:, :3]
        close3 = np.nanmean(np.take_along_axis(closing, order, axis=1), axis=1)
    return {"def_line": def_line, "mid_line": mid_line, "width": width, "depth": depth,
            "centroid_x": cx, "compactness": compact, "near_ball": near, "closing": close3}


# Standard shapes, read from the back line forward (goalkeeper excluded): "4-3-3" = 4 defenders, 3 midfielders, 3 forwards.
FORMATIONS = {
    10: ["4-4-2", "4-3-3", "4-2-3-1", "4-1-4-1", "4-4-1-1", "4-3-1-2", "4-1-3-2", "4-2-2-2", "4-5-1",
         "3-5-2", "3-4-3", "3-4-2-1", "3-4-1-2", "3-3-2-2", "5-3-2", "5-4-1"],
    9: ["4-4-1", "4-3-2", "4-2-3", "3-4-2", "3-5-1", "5-3-1", "4-2-2-1", "4-3-1-1"],
    8: ["4-3-1", "4-2-2", "3-4-1", "3-3-2"],
}
_SPAN = {1: 0.0, 2: 0.45, 3: 0.7, 4: 0.9, 5: 1.0, 6: 1.0}


def _template(name, with_lines=False):
    lines = [int(c) for c in name.split("-")]
    pts, line_of = [], []
    for i, n in enumerate(lines):
        x = i / (len(lines) - 1)
        pts += [(x, y) for y in np.linspace(-1, 1, n) * _SPAN[n]]
        line_of += [i] * n
    return (np.array(pts), np.array(line_of)) if with_lines else np.array(pts)


def formation_costs(avg):
    """{formation: fit cost per player} for every standard shape with len(avg) outfield players."""
    from scipy.optimize import linear_sum_assignment
    avg = np.asarray(avg, dtype=float)
    out = {}
    names = FORMATIONS.get(len(avg))
    if not names:
        return out
    z = (avg - avg.mean(0)) / (avg.std(0) + 1e-6)
    for name in names:
        tp = _template(name)
        tz = (tp - tp.mean(0)) / (tp.std(0) + 1e-6)
        fit = tz.copy()
        for _ in range(6):
            r, c = linear_sum_assignment(((z[:, None, :] - fit[None, :, :]) ** 2).sum(-1))
            fit = tz.copy()
            for ax in range(2):
                a, b = tz[c, ax], z[r, ax]
                sc, off = np.linalg.lstsq(np.vstack([a, np.ones_like(a)]).T, b, rcond=None)[0]
                fit[:, ax] = max(sc, 0.3) * tz[:, ax] + off
        out[name] = float(((z[r] - fit[c]) ** 2).sum()) / len(avg)
    return out


SWITCH_COST = 0.10  # per-player fit cost saving needed to change shape (real DFL: changes per team-half 6.9 -> 1.1)


def smooth_formations(costs):
    """Viterbi over windows: the formation sequence minimising fit cost + SWITCH_COST per change.
    costs: list of {name: cost} (empty dict = unknown window). Returns list of names (or "?")."""
    names = sorted({n for c in costs for n in c})
    if not names:
        return ["?"] * len(costs)
    INF = 1e9
    prev = {n: 0.0 for n in names}
    back = []
    for c in costs:
        cur, bp = {}, {}
        for n in names:
            stay = prev[n]
            best_other = min(((prev[o] + SWITCH_COST, o) for o in names if o != n), default=(INF, None))
            base, frm = (stay, n) if stay <= best_other[0] else best_other
            cur[n] = base + (c.get(n, INF) if c else 0.0)
            bp[n] = frm
        back.append(bp)
        prev = cur
    n = min(prev, key=prev.get)
    seq = [n]
    for bp in reversed(back[1:]):
        n = bp[n]
        seq.append(n)
    seq = seq[::-1]
    return [s if c and s in c else ("?" if not c else min(c, key=c.get)) for s, c in zip(seq, costs)]


def fit_formation(avg, with_lines=False):
    """avg: (n, 2) mean outfield positions in the team's own frame (x towards the opponent goal).
    Fits every standard shape for n players (assignment + per-axis scale/offset) and returns the best name
    (and, with_lines, the player indices in each line from the back line forward)."""
    from scipy.optimize import linear_sum_assignment
    avg = np.asarray(avg, dtype=float)
    n = len(avg)
    names = FORMATIONS.get(n)
    if not names:
        return ("?", []) if with_lines else "?"
    z = (avg - avg.mean(0)) / (avg.std(0) + 1e-6)
    best, best_cost, best_lines = "?", np.inf, []
    for name in names:
        tp = _template(name)
        tz = (tp - tp.mean(0)) / (tp.std(0) + 1e-6)
        fit = tz.copy()
        for _ in range(6):
            r, c = linear_sum_assignment(((z[:, None, :] - fit[None, :, :]) ** 2).sum(-1))
            fit = tz.copy()
            for ax in range(2):  # least-squares scale + offset per axis (scale kept positive: defence stays behind)
                a, b = tz[c, ax], z[r, ax]
                A = np.vstack([a, np.ones_like(a)]).T
                sc, off = np.linalg.lstsq(A, b, rcond=None)[0]
                fit[:, ax] = max(sc, 0.3) * tz[:, ax] + off
        cost = ((z[r] - fit[c]) ** 2).sum()
        if cost < best_cost:
            _, line_of = _template(name, True)
            best, best_cost = name, cost
            best_lines = [[int(i) for i, cc in zip(r, c) if line_of[cc] == k] for k in range(line_of.max() + 1)]
    return (best, best_lines) if with_lines else best


def window_average(m: MatchData, team: str, t0: int, t1: int):
    """(player ids, mean outfield positions relative to the team centroid) over [t0, t1), own frame."""
    sel = np.arange(max(0, t0), min(m.T, t1))
    mask = m.team_mask(team)
    p = m.own_xy(team)[sel][:, mask]
    with np.errstate(all="ignore"), np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        rel = p - np.nanmean(p, axis=1, keepdims=True)
        present = np.mean(~np.isnan(rel[..., 0]), axis=0) > 0.5
        avg = np.nanmean(rel, axis=0)[present]
    pids = [pid for pid, keep in zip([pid for pid, k in zip(m.order, mask) if k], present) if keep]
    return pids, avg


def detect_formation(m: MatchData, team: str, t0: int, t1: int, with_lines=False):
    """Mean positions relative to the team centroid over the window (both phases: on real DFL tracking this
    matched the official line-up better than out-of-possession only), then the best-fitting standard shape."""
    sel = np.arange(max(0, t0), min(m.T, t1))
    mask = m.team_mask(team)
    p = m.own_xy(team)[sel][:, mask]
    with np.errstate(all="ignore"), np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        rel = p - np.nanmean(p, axis=1, keepdims=True)
        present = np.mean(~np.isnan(rel[..., 0]), axis=0) > 0.5
        avg = np.nanmean(rel, axis=0)[present]
    if len(avg) < 8:
        return ("?", []) if with_lines else "?"
    if not with_lines:
        return fit_formation(avg)
    name, lines = fit_formation(avg, True)
    pids = [pid for pid, keep in zip([pid for pid, k in zip(m.order, mask) if k], present) if keep]
    return name, [[pids[i] for i in ln] for ln in lines]


STICKY = 0.6  # extra cost (z-units squared) for moving a player to a different line than the previous second


def assign_lines(pts, name, prev=None):
    """Assign current outfield positions (n, 2; own frame) to the lines of formation `name`, back line first.
    Same fit as fit_formation, with a penalty for changing a player's line from the previous second (prev)."""
    from scipy.optimize import linear_sum_assignment
    pts = np.asarray(pts, dtype=float)
    try:
        tp, line_of = _template(name, True)
    except (ValueError, KeyError):
        return None
    if len(tp) != len(pts):
        return None
    z = (pts - pts.mean(0)) / (pts.std(0) + 1e-6)
    tz = (tp - tp.mean(0)) / (tp.std(0) + 1e-6)
    fit = tz.copy()
    stick = np.zeros((len(pts), len(tp)))
    if prev is not None:
        stick = STICKY * (np.asarray(prev)[:, None] != line_of[None, :])
    for _ in range(4):
        r, c = linear_sum_assignment(((z[:, None, :] - fit[None, :, :]) ** 2).sum(-1) + stick)
        fit = tz.copy()
        for ax in range(2):
            a, b = tz[c, ax], z[r, ax]
            sc, off = np.linalg.lstsq(np.vstack([a, np.ones_like(a)]).T, b, rcond=None)[0]
            fit[:, ax] = max(sc, 0.3) * tz[:, ax] + off
    out = np.empty(len(pts), dtype=int)
    out[r] = line_of[c]
    return out


def lines_per_second(frames, order, is_gk, timeline):
    """Per frame and team: line index of every player (-1 = goalkeeper / off / unknown), from the formation of
    that minute fitted to the current positions."""
    out = []
    prev = {"A": (None, None), "B": (None, None)}
    for t, f in enumerate(frames):
        row = [-1] * len(order)
        for team in ("A", "B"):
            tl = timeline[team]
            name = tl[min(t // 60, len(tl) - 1)].get("formation") if tl else None
            js = [j for j, pid in enumerate(order) if pid.startswith(team) and not is_gk[j] and f["xy"][j] is not None]
            if not name or name == "?" or not js:
                continue
            p = np.array([f["xy"][j] for j in js], dtype=float)
            if team == "B":
                p = np.stack([L - p[:, 0], W - p[:, 1]], 1)
            pj, pl = prev[team]
            pr = [pl[pj.index(j)] if pj is not None and j in pj else -1 for j in js] if pj is not None else None
            lab = assign_lines(p, name, pr)
            if lab is None:
                prev[team] = (None, None)
                continue
            for j, k in zip(js, lab):
                row[j] = int(k)
            prev[team] = (js, list(lab))
        out.append(row)
    return out
