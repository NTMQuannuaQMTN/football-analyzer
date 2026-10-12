"""Real-data benchmark: DFL full-pitch ground truth, masked like a broadcast camera.

Masks
  sc_camera   real SkillCorner broadcast footprints (pan/zoom/cuts + real detection misses), translated onto the
              DFL ball position (camera follows the ball) — the realistic setting
  k=<n>       the n players nearest the (ball-following) camera centre are visible: 18/16/14/12/10/8/6
  rand=<n>    n players visible chosen uniformly at random (secondary robustness benchmark)
Baselines
  last_position, constant_velocity, formation (team-shape estimator, live / replay)
Split (match level; Fortuna Düsseldorf plays in 5 of 7 matches, so test teams are never seen in training)
  train: the 5 Düsseldorf matches   val/test: Köln-Bayern and Bochum-Leverkusen (2 folds, swapped)
"""
import json
from types import SimpleNamespace

import numpy as np

from backend.reconstruction.datasets import CACHE, L, W, dfl_matches, parse_dfl, point_in_quad, skillcorner_cameras

UNSEEN = ["DFL-MAT-J03WMX", "DFL-MAT-J03WN1"]  # Köln-Bayern, Bochum-Leverkusen
LEVELS_K = [18, 16, 14, 12, 10, 8, 6]

# DFL playing position -> (line, lateral); lateral +1 = right flank. In DFL own frame the right flank is LOW y.
DFL_LINE_LAT = {"TW": (0, 0), "IVL": (1, -.3), "IVR": (1, .3), "IVZ": (1, 0), "LV": (1, -.85), "RV": (1, .85),
                "DML": (1.6, -.3), "DMR": (1.6, .3), "DMZ": (1.6, 0), "DLM": (2, -.4), "DRM": (2, .4), "ZM": (2, 0),
                "ZO": (2.45, 0), "OLM": (2.2, -.8), "ORM": (2.2, .8), "LM": (2, -.85), "RM": (2, .85),
                "LA": (2.6, -.8), "RA": (2.6, .8), "HL": (1.5, -.9), "HR": (1.5, .9), "STZ": (3.1, 0), "STL": (3, -.25),
                "STR": (3, .25), "HST": (2.8, 0)}


def split(fold=0):
    others = [m for m in dfl_matches() if m not in UNSEEN]
    test, val = (UNSEEN[0], UNSEEN[1]) if fold == 0 else (UNSEEN[1], UNSEEN[0])
    return others, val, test


def segment_view(info, seg):
    """MatchData-like view of one DFL half for the team-shape estimator."""
    if seg.get('sampling_hz',1)!=1 or 'timestamps_s' in seg and not np.allclose(np.diff(seg['timestamps_s']),1.,atol=1e-6):
        raise ValueError('Current reconstruction features require uniform 1 Hz tracking; resample before benchmarking')
    order = [f"{t}:{p}" for t, p in zip(seg["team"], seg["pids"])]
    on = ~np.isnan(seg["pos"][..., 0])
    role = {o: ("GK" if r == "GK" else "CM") for o, r in zip(order, seg["role"])}
    anchors = {}
    for o, pid in zip(order, seg["pids"]):
        ln, lat = DFL_LINE_LAT.get(info["players"][pid]["dfl_position"] or "", (2, 0))
        anchors[o] = ((ln - 1) / 2 * 30 - 15, -lat * 22)  # right flank = low y in the DFL frame
    m = SimpleNamespace(xy=seg["pos"], ball=seg["ball"], poss=seg["poss"], order=order, T=len(seg["pos"]),
                        idx={o: j for j, o in enumerate(order)}, players=[], events=[])
    return m, on, role, anchors


# ----------------------------------------------------------------------------- masks
def camera_centre(ball_x):
    """Causal simulated camera pan: initialise from the current ball, never future balls."""
    cam = np.empty_like(ball_x)
    c = float(ball_x[0]) if len(ball_x) and np.isfinite(ball_x[0]) else L / 2
    for t, bx in enumerate(ball_x):
        if np.isfinite(bx):
            c += 0.5 * (bx - c)
        cam[t] = np.clip(c, 22, L - 22)
    return cam


def mask_k(seg, k):
    on = ~np.isnan(seg["pos"][..., 0])
    cam = camera_centre(seg["ball"][:, 0])
    by = np.nan_to_num(seg["ball"][:, 1], nan=W / 2)
    d = np.hypot(seg["pos"][..., 0] - cam[:, None], (seg["pos"][..., 1] - by[:, None]) * 0.5)
    d[~on] = np.inf
    vis = np.zeros(on.shape, bool)
    order = np.argsort(d, 1)[:, :k]
    np.put_along_axis(vis, order, True, 1)
    return vis & on


def mask_random(seg, k, seed=0):
    rng = np.random.default_rng(seed)
    on = ~np.isnan(seg["pos"][..., 0])
    r = rng.random(on.shape)
    r[~on] = np.inf
    vis = np.zeros(on.shape, bool)
    np.put_along_axis(vis, np.argsort(r, 1)[:, :k], True, 1)
    return vis & on


def mask_sc(seg, seed=0, cams=None, return_quads=False):
    """Transfer real SkillCorner camera footprints: corners(t') - ball_sc(t') + ball_dfl(t); cuts and real
    in-footprint detection rates are kept. return_quads: also the per-second footprint (T, 4, 2), NaN on cuts."""
    cams = cams or skillcorner_cameras()
    rng = np.random.default_rng(seed)
    T = len(seg["pos"])
    on = ~np.isnan(seg["pos"][..., 0])
    # concatenate a random sequence of SkillCorner periods until we cover T seconds
    order = rng.permutation(len(cams))
    corners, bsc, rate = [], [], []
    i = 0
    while sum(len(c) for c in corners) < T:
        c = cams[order[i % len(cams)]]
        start = rng.integers(0, max(1, len(c["corners"]) - 60))
        corners.append(c["corners"][start:])
        bsc.append(c["ball"][start:])
        rate.append(c["det_rate"][start:])
        i += 1
    corners = np.concatenate(corners)[:T]
    bsc = np.concatenate(bsc)[:T]
    rate = np.concatenate(rate)[:T]
    vis = np.zeros(on.shape, bool)
    quads = np.full((T, 4, 2), np.nan)
    bd = seg["ball"]
    for t in range(T):
        q = corners[t]
        if np.isnan(q).any():
            continue  # camera cut / no footprint: nobody visible
        if np.isfinite(bsc[t]).all() and np.isfinite(bd[t]).all():
            q = q - bsc[t] + bd[t]
        quads[t] = q
        inside = point_in_quad(np.nan_to_num(seg["pos"][t], nan=-999), q)
        p = rate[t] if np.isfinite(rate[t]) else 0.9
        vis[t] = inside & on[t] & (rng.random(on.shape[1]) < p)
    return (vis, quads) if return_quads else vis


# ----------------------------------------------------------------------------- baselines
def last_position(seg, vis):
    pos = np.full(seg["pos"].shape, np.nan)
    last = np.full(seg["pos"].shape[1:], np.nan)
    for t in range(len(pos)):
        last = np.where(vis[t][:, None], seg["pos"][t], last)
        pos[t] = last
    return pos


def constant_velocity(seg, vis, tau=2.0):
    """Last position + last observed velocity, decaying (players don't run in straight lines forever)."""
    T, N = vis.shape
    pos = np.full(seg["pos"].shape, np.nan)
    last = np.full((N, 2), np.nan)
    vel = np.zeros((N, 2))
    t_last = np.full(N, -1)
    for t in range(T):
        seen = vis[t]
        if t > 0:
            cont = seen & vis[t - 1]
            vel[cont] = seg["pos"][t, cont] - seg["pos"][t - 1, cont]
            vel[seen & ~vis[t - 1]] = 0
        last[seen] = seg["pos"][t, seen]
        t_last[seen] = t
        dt = (t - t_last)[:, None]
        pos[t] = last + vel * tau * (1 - np.exp(-dt / tau))
    return np.clip(pos, [0, 0], [L, W])


def formation(info, seg, vis, mode, calib=None):
    from backend.reconstruction.estimator import reconstruct
    m, on, role, anchors = segment_view(info, seg)
    return reconstruct(m, vis & on, mode, calib if calib is not None else {}, on=on, role=role, anchors=anchors)


# ----------------------------------------------------------------------------- metrics
def errors(seg, vis, pred):
    on = ~np.isnan(seg["pos"][..., 0])
    hid = on & ~vis & np.isfinite(pred[..., 0])
    e = np.linalg.norm(pred - seg["pos"], axis=2)
    return e, hid


def summarize(e):
    e = np.asarray(e)
    if not len(e):
        return None
    return {"n": int(len(e)), "mae_m": round(float(e.mean()), 2), "rmse_m": round(float(np.sqrt((e ** 2).mean())), 2),
            "within_5m": round(float((e <= 5).mean()), 3), "within_10m": round(float((e <= 10).mean()), 3)}


def breakdown(seg, vis, e, hid):
    """Error by role and by tactical state (possession phase of the player's team, ball third)."""
    out = {"by_role": {}, "by_state": {}}
    roles = seg["role"]
    for r in sorted(set(roles)):
        sel = hid & (roles == r)[None]
        if sel.sum() > 200:
            out["by_role"][r] = summarize(e[sel])
    team = seg["team"]
    poss = seg["poss"]
    bx = seg["ball"][:, 0]
    for name, cond in {
        "in_possession": (poss[:, None] == np.where(team == "A", 0, 1)[None]),
        "out_of_possession": (poss[:, None] == np.where(team == "A", 1, 0)[None]),
        "ball_dead": ~seg["alive"][:, None].repeat(len(team), 1),
    }.items():
        sel = hid & cond
        if sel.sum() > 200:
            out["by_state"][name] = summarize(e[sel])
    own_bx = np.where(team == "A", bx[:, None], L - bx[:, None])
    for name, (a, b) in {"ball_in_own_third": (0, 35), "ball_in_middle_third": (35, 70), "ball_in_attacking_third": (70, 106)}.items():
        sel = hid & (own_bx >= a) & (own_bx < b)
        if sel.sum() > 200:
            out["by_state"][name] = summarize(e[sel])
    return out


def run_baselines(matches, masks=("sc_camera", "k=14", "k=10", "k=6"), seed=0, calib=None):
    """Error arrays per mask and method over the given matches (concatenated)."""
    res = {}
    for mid in matches:
        info, segs = parse_dfl(mid)
        for si, seg in enumerate(segs):
            for mk in masks:
                vis = make_mask(seg, mk, seed + si)
                preds = {"last_position": last_position(seg, vis), "constant_velocity": constant_velocity(seg, vis)}
                for mode in ("live", "replay"):
                    preds[f"formation_{mode}"] = formation(info, seg, vis, mode, calib).pos
                for name, p in preds.items():
                    e, hid = errors(seg, vis, p)
                    res.setdefault(mk, {}).setdefault(name, []).append(e[hid])
    return {mk: {n: summarize(np.concatenate(v)) for n, v in d.items()} for mk, d in res.items()}


def make_mask(seg, mk, seed=0):
    if mk == "sc_camera":
        return mask_sc(seg, seed)
    if mk.startswith("k="):
        return mask_k(seg, int(mk[2:]))
    if mk.startswith("rand="):
        return mask_random(seg, int(mk[5:]), seed)
    raise ValueError(mk)


if __name__ == "__main__":
    tr, va, te = split(0)
    print(json.dumps(run_baselines([te]), indent=1))
