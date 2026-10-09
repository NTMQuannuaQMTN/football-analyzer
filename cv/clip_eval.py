"""Prediction -> reality on REAL broadcast video (SoccerNet clip, CV pipeline output). No ground truth exists for
off-screen players, but players do come back into view: for every reappearance after >= 5 s off screen, each
method predicts (causally, with that second masked) where the player is; the "actual" position is where the video
then shows him. Caveats: the actual position is itself a CV detection, and player identity in the clip is
role-based (about 35% exact), so absolute errors are inflated; the comparison between methods is on identical cases.

    .venv/bin/python -m cv.clip_eval --models f0 k7r_all_v3
"""
import argparse
import json
from collections import Counter
from types import SimpleNamespace

import numpy as np

from backend.reconstruction.estimator import reconstruct
from backend.reconstruction.inference import ml_available, reconstruct_ml

CLIP = "data/cv_2016_08_14_18_00_arsenal_3_4_liverpool_h1_1440_2040.json"
MIN_HIDDEN = 5


def coarse_roles(players):
    """Coarse role per player from the lineup slot (L<line><k>) and its line size."""
    lines = Counter((p["team"], p["role_slot"][:2]) for p in players if p["role_slot"] != "GK")
    last_line = {tm: max(int(p["role_slot"][1]) for p in players if p["team"] == tm and p["role_slot"] != "GK") for tm in "AB"}
    out = {}
    for p in players:
        s = p["role_slot"]
        if s == "GK":
            out[p["id"]] = "GK"
            continue
        ln, k = int(s[1]), int(s[2])
        n = lines[(p["team"], s[:2])]
        outer = k in (1, n) and n >= 3
        if ln == 1:
            out[p["id"]] = "FB" if outer else "CB"
        elif ln == last_line[p["team"]]:
            out[p["id"]] = "W" if outer else "ST"
        elif ln == last_line[p["team"]] - 1 and ln > 2:
            out[p["id"]] = "W" if outer else "AM"
        else:
            out[p["id"]] = ("WM" if n >= 4 else "CM") if outer else ("DM" if n <= 2 else "CM")
    return out


def load(path=CLIP):
    d = json.load(open(path))
    F = d["frames"]
    order = d["meta"]["player_order"]
    T, N = len(F), len(order)
    xy = np.full((T, N, 2), np.nan)
    vis = np.zeros((T, N), bool)
    on = np.zeros((T, N), bool)
    for t, f in enumerate(F):
        for j, (p, o) in enumerate(zip(f["xy"], f["obs"])):
            if o is not None:
                on[t, j] = True
            if o is True and p is not None:
                xy[t, j], vis[t, j] = p, True
    ball = np.array([f["b"] for f in F], dtype=float)
    poss = np.array([{"A": 0, "B": 1}.get(f["p"], -1) for f in F])
    m = SimpleNamespace(xy=xy, ball=ball, poss=poss, order=order, T=T, idx={o: j for j, o in enumerate(order)},
                        players=d["players"], events=[])
    return d, m, vis, on


def reveals(vis, on):
    out = []
    for j in range(vis.shape[1]):
        hidden = 0
        for t in range(vis.shape[0]):
            if not on[t, j]:
                hidden = 0
            elif vis[t, j]:
                if hidden >= MIN_HIDDEN and t >= hidden:
                    out.append((t, j, hidden))
                hidden = 0
            else:
                hidden += 1
    return out


def last_position(m, vis):
    pos = np.full(m.xy.shape, np.nan)
    last = np.full(m.xy.shape[1:], np.nan)
    for t in range(m.T):
        last = np.where(vis[t][:, None], m.xy[t], last)
        pos[t] = last
    return pos


def main(models):
    d, m, vis, on = load()
    rv = reveals(vis, on)
    masked = vis.copy()
    for t, j, _ in rv:
        masked[t, j] = False
    teams = np.array([o[0] for o in m.order])
    cr = coarse_roles(d["players"])
    rc = np.array([cr[o] for o in m.order])
    role = {o: ("GK" if cr[o] == "GK" else "CM") for o in m.order}
    preds = {"last_position": (last_position(m, masked), None),
             "team_shape_live": (reconstruct(m, masked, "live", {}, on=on, role=role).pos, None)}
    for name in models:
        try:
            r = reconstruct_ml(m, masked, "live", on, role, None, teams, rc, fold=name)
            preds[f"learned_{name}"] = (r.pos, r.rad)
        except FileNotFoundError:
            print(f"(no weights for {name})")
    res = {"clip": d["meta"]["source"]["game"], "reappearances": len(rv), "min_hidden_s": MIN_HIDDEN,
           "caveat": "actual = CV detection at reappearance; identities role-based (~35% exact)", "methods": {}}
    for name, (p, rad) in preds.items():
        e = np.array([np.hypot(*(p[t, j] - m.xy[t, j])) for t, j, _ in rv])
        ok = np.isfinite(e)
        row = {"n": int(ok.sum()), "median_error_m": round(float(np.median(e[ok])), 2), "mean_error_m": round(float(e[ok].mean()), 2),
               "within_10m": round(float((e[ok] <= 10).mean()), 3)}
        if rad is not None:
            r = np.array([rad[t, j] for t, j, _ in rv])
            row["inside_80pct_radius"] = round(float((e[ok] <= r[ok]).mean()), 3)
        res["methods"][name] = row
        print(f"{name:22s} {row}")
    # video-domain calibration: one radius scale fitted on the first half of the clip's reappearances, coverage
    # checked on the second half (time split, no overlap)
    cal = {}
    mid_t = m.T // 2
    for name, (p, rad) in preds.items():
        if rad is None:
            continue
        rows = [(t, np.hypot(*(p[t, j] - m.xy[t, j])), rad[t, j]) for t, j, _ in rv]
        fit = np.array([(e, r) for t, e, r in rows if t < mid_t and np.isfinite(e) and r > 0])
        test = np.array([(e, r) for t, e, r in rows if t >= mid_t and np.isfinite(e) and r > 0])
        if len(fit) < 20 or len(test) < 20:
            continue
        scale = float(np.quantile(fit[:, 0] / fit[:, 1], 0.8))
        cal[name] = {"radius_scale": round(scale, 2), "fit_n": len(fit), "test_n": len(test),
                     "test_coverage_before": round(float((test[:, 0] <= test[:, 1]).mean()), 3),
                     "test_coverage_after": round(float((test[:, 0] <= scale * test[:, 1]).mean()), 3)}
        print(f"video calibration {name}: {cal[name]}")
    res["video_calibration"] = cal
    json.dump(res, open("data/eval/real/clip_reveals.json", "w"), indent=1)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["f0"])
    a = ap.parse_args()
    main(a.models)
