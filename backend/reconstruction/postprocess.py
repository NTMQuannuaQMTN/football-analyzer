"""Post-processing for hidden-player estimates (applied after the formation prior + learned correction).

  event_pins     a player named in the event feed (passer, shooter, tackle winner) is at the ball at that moment
  shape_labels   each team's current shape (best-fitting standard formation over a window of estimates)
  context_blend  "where does this player usually stand in this shape and phase?": his mean offset from the team
                 centroid over the seconds he was seen with the same (shape, in/out of possession); replay uses every
                 sighting, live only earlier ones
  limit_speed    nobody teleports: consecutive estimates differ by at most VMAX m/s. Replay smooths each hidden
                 spell and pulls it towards both the last and the next sighting; live only constrains forward in time
"""
import numpy as np

from backend.analytics.shape import fit_formation

VMAX = 8.5          # m/s, sustained sprint (estimates only; observed positions are never moved)
SHAPE_WINDOW = 300  # s
SHAPE_STEP = 60     # s


def _ball_at(ball, t):
    i = int(t)
    a = t - i
    return ball[i] * (1 - a) + ball[min(len(ball) - 1, i + 1)] * a


def estimate_lag(events, ball, col, vis, pos, kinds=("PASS", "SHOT", "TACKLE")):
    """Event-clock offset (s) from events whose player is on camera: argmin of ball-to-player distance.
    Uses only observed positions, so it is available in a real broadcast too."""
    T = len(ball)
    best, best_d = 0.0, np.inf
    for lag in np.arange(-3.0, 3.01, 0.25):
        d = []
        for e in events:
            if e["event_type"] not in kinds or e.get("player_id") not in col:
                continue
            t = e["timestamp"] + lag
            k, j = int(round(t)), col[e["player_id"]]
            if 0 <= t < T - 1 and vis[k, j]:
                d.append(np.hypot(*(_ball_at(ball, t) - pos[k, j])))
        d = np.array(d)
        d = d[np.isfinite(d)]
        if len(d) >= 30 and np.median(d) < best_d:
            best, best_d = float(lag), float(np.median(d))
    return best


def event_pins(events, ball, col, on, kinds=("PASS", "SHOT", "GOAL", "TACKLE"), lag=0.0):
    """{(t, j): (x, y)}: the acting player is at the ball (interpolated to the event time, put on the nearest second)."""
    T = len(ball)
    pins = {}
    for e in events:
        if e["event_type"] not in kinds or e.get("player_id") not in col:
            continue
        t = e["timestamp"] + lag
        k = int(round(t))
        if not 0 <= k < T:
            continue
        a, i = t - int(t), int(t)
        xy = ball[i] * (1 - a) + ball[min(T - 1, i + 1)] * a
        j = col[e["player_id"]]
        if np.isfinite(xy).all() and on[k, j]:
            pins[(k, j)] = (float(xy[0]), float(xy[1]))
    return pins


def shape_labels(pos, team, roles, on, causal):
    """Per second, the team's shape from the estimates of the surrounding (replay) or preceding (live) window."""
    T = len(pos)
    cols = [j for j in range(pos.shape[1]) if team[j] == "A" or team[j] == "B"]
    out = {}
    for tm in ("A", "B"):
        js = [j for j in cols if team[j] == tm and roles[j] != "GK"]
        lab = np.empty(T, dtype=object)
        for s in range(0, T, SHAPE_STEP):
            a, b = (max(0, s + SHAPE_STEP - SHAPE_WINDOW), s + SHAPE_STEP) if causal else \
                   (max(0, s + SHAPE_STEP // 2 - SHAPE_WINDOW // 2), s + SHAPE_STEP // 2 + SHAPE_WINDOW // 2)
            sel = [j for j in js if on[a:b, j].mean() > 0.5] if b > a else []
            p = pos[a:b][:, sel]
            if tm == "B":
                p = np.stack([105.0 - p[..., 0], 68.0 - p[..., 1]], -1)
            with np.errstate(all="ignore"):
                rel = np.nanmean(p - np.nanmean(p, 1, keepdims=True), 0) if len(sel) >= 8 else None
            lab[s:s + SHAPE_STEP] = fit_formation(rel) if rel is not None and np.isfinite(rel).all() else "?"
        out[tm] = lab
    return out


def context_blend(pos, fixed, team, roles, on, poss, w, causal):
    """Blend hidden estimates with the player's usual offset from his team centroid in the current (shape, phase)."""
    if w <= 0:
        return pos
    T, N, _ = pos.shape
    out = pos.copy()
    labels = shape_labels(pos, team, roles, on, causal)
    for tm, code in (("A", 0), ("B", 1)):
        js = [j for j in range(N) if team[j] == tm]
        out_js = [j for j in js if roles[j] != "GK"]
        with np.errstate(all="ignore"):
            cen = np.nanmean(np.where(on[:, out_js, None], pos[:, out_js], np.nan), 1)  # (T, 2)
        phase = np.where(poss == code, "in", np.where(poss < 0, "dead", "out"))
        key = np.array([f"{l}|{p}" for l, p in zip(labels[tm], phase)])
        for j in js:
            off = pos[:, j] - cen
            seen = fixed[:, j] & on[:, j] & np.isfinite(off).all(1)
            hid = on[:, j] & ~fixed[:, j]
            if not hid.any() or not seen.any():
                continue
            est = np.full((T, 2), np.nan)
            if causal:
                sums, cnt = {}, {}
                for t in range(T):
                    if hid[t] and cnt.get(key[t], 0) >= 5:
                        est[t] = sums[key[t]] / cnt[key[t]]
                    if seen[t]:
                        sums[key[t]] = sums.get(key[t], 0) + off[t]
                        cnt[key[t]] = cnt.get(key[t], 0) + 1
            else:
                for k in np.unique(key[hid]):
                    s = seen & (key == k)
                    if s.sum() >= 5:
                        est[hid & (key == k)] = off[s].mean(0)
            ok = hid & np.isfinite(est).all(1)
            out[ok, j] = (1 - w) * pos[ok, j] + w * (cen[ok] + est[ok])
    return out


def limit_speed(pos, fixed, on, causal, vmax=VMAX, iters=6, sigma=1.5):
    """Speed-limit hidden estimates; observed / pinned positions (fixed) are anchors and never move."""
    T, N, _ = pos.shape
    p = pos.copy()
    free = on & ~fixed
    if not causal:  # light smoothing inside hidden spells only
        k = np.exp(-0.5 * (np.arange(-4, 5) / sigma) ** 2)
        for j in range(N):
            v = np.where(on[:, j, None], p[:, j], 0.0)
            wsum = np.convolve(on[:, j].astype(float), k, "same")
            sm = np.stack([np.convolve(v[:, a], k, "same") for a in range(2)], 1) / np.maximum(wsum, 1e-9)[:, None]
            p[free[:, j], j] = sm[free[:, j]]

    def clamp(t, s):  # move p[t] towards p[s] so |p[t] - p[s]| <= vmax, for free players on in both
        m = free[t] & on[s]
        if not m.any():
            return
        d = p[t, m] - p[s, m]
        r = np.linalg.norm(d, axis=1)
        over = r > vmax
        if over.any():
            idx = np.where(m)[0][over]
            p[t, idx] = p[s, idx] + d[over] / r[over, None] * vmax

    for _ in range(1 if causal else iters):
        for t in range(1, T):
            clamp(t, t - 1)
        if causal:
            break
        for t in range(T - 2, -1, -1):
            clamp(t, t + 1)
    return p


def jump_stats(pos, on, mask=None):
    """Displacement between consecutive seconds (m), over players on in both seconds (optionally only where mask)."""
    d = np.linalg.norm(np.diff(pos, axis=0), axis=2)
    ok = on[1:] & on[:-1] & np.isfinite(d)
    if mask is not None:
        ok &= mask[1:] | mask[:-1]
    d = d[ok]
    return {"p99_m_per_s": round(float(np.percentile(d, 99)), 1), "max_m_per_s": round(float(d.max()), 1),
            "over_10_m_per_s": int((d > 10).sum()), "steps": int(len(d))}
