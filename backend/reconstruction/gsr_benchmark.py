"""Real-broadcast hidden-player benchmark on SoccerNet-GSR (external to every DFL-trained model).

SoccerNet-GSR clips are 30 s of real broadcast video (25 fps) with, for every person IN FRAME, a pitch position,
team side, role, jersey and a clip-consistent track id. Off-screen players have no label, so hidden-player error
can only be measured where truth exists:

  live re-entry   a player leaves the frame and re-enters; at the re-entry second the player is treated as hidden
                  and the LIVE estimate (past only) is compared with the labelled re-entry position.
                  The visibility pattern and the prior trajectory are 100 % real broadcast.
  replay extend   the same re-entries, but the first EXT s after re-entry are additionally hidden, so the REPLAY
                  estimate (using the later sighting) can be scored there. Partly simulated (EXT s of extra hiding).

Inputs to every method are the labelled visible positions (oracle perception and oracle track identity). This
isolates the hidden-player model from detection / calibration / identity errors, which are measured separately.
Domain gaps vs DFL: unknown positional roles (only GK vs outfield), no possession / ball-status feed (set to
unknown / in play), ball position only when labelled in frame, and only players seen at least once in the clip.

    .venv/bin/python -m backend.reconstruction.gsr_benchmark --methods base,final_v3,r13_mlp_final
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from backend.reconstruction import hidden_models as hm
from backend.reconstruction.datasets import L, W
from backend.reconstruction.estimator import reconstruct
from backend.reconstruction.ml import apply_calibration, features
from backend.reconstruction.real_benchmark import constant_velocity, last_position

ROOT = Path(__file__).resolve().parents[2]
GSR = ROOT / "data" / "sn_gsr"
OUT = ROOT / "data" / "eval" / "recon_benchmark"
EXT = 3       # s of extra hiding after a real re-entry for the replay score
FPS_STEP = 25  # 1 Hz from 25 fps


def load_clip(path):
    d = json.load(open(path / "Labels-GameState.json"))
    info = d["info"]
    images = sorted(d["images"], key=lambda im: im["image_id"])
    keep = {im["image_id"]: k // FPS_STEP for k, im in enumerate(images) if k % FPS_STEP == 0}
    T = max(keep.values()) + 1
    tracks, ball = {}, np.full((T, 2), np.nan)
    for a in d["annotations"]:
        if a["supercategory"] != "object" or a["image_id"] not in keep:
            continue
        t = keep[a["image_id"]]
        p = a.get("bbox_pitch") or {}
        x, y = p.get("x_bottom_middle"), p.get("y_bottom_middle")
        if x is None or abs(x) > 60 or abs(y) > 40:
            continue
        xy = (x + L / 2, y + W / 2)
        at = a["attributes"]
        if at.get("role") == "ball":
            ball[t] = xy
        elif at.get("role") in ("player", "goalkeeper") and at.get("team") in ("left", "right"):
            k = a["track_id"]
            tr = tracks.setdefault(k, {"team": "A" if at["team"] == "left" else "B",
                                       "role": "GK" if at["role"] == "goalkeeper" else "CM", "pos": {}})
            tr["pos"][t] = xy
    ids = sorted(tracks)
    pos = np.full((T, len(ids), 2), np.nan)
    for j, k in enumerate(ids):
        for t, xy in tracks[k]["pos"].items():
            pos[t, j] = xy
    ball_seen = np.isfinite(ball).all(1)
    last = np.array([L / 2, W / 2])
    for t in range(T):  # causal forward fill: last labelled ball, pitch centre before the first one
        if ball_seen[t]:
            last = ball[t].copy()
        else:
            ball[t] = last
    team = np.array([tracks[k]["team"] for k in ids])
    role = np.array([tracks[k]["role"] for k in ids])
    return {"clip": info["name"], "game": info["game_id"], "action": info.get("action_class"), "pos": pos, "ball": ball,
            "team": team, "role": role, "poss": np.full(T, -1), "alive": np.ones(T, bool)}


def reentries(vis):
    """(t, j, gap_s) where j was visible before, hidden for gap_s samples, visible again at t."""
    T, N = vis.shape
    out = []
    for j in range(N):
        seen, gap = False, 0
        for t in range(T):
            if vis[t, j]:
                if seen and gap > 0:
                    out.append((t, j, gap))
                seen, gap = True, 0
            elif seen:
                gap += 1
    return out


def estimate(clip, vis, mode, model=None):
    T, N = vis.shape
    order = [f"{tm}:{j}" for j, tm in enumerate(clip["team"])]
    m = SimpleNamespace(xy=np.where(vis[..., None], clip["pos"], np.nan), ball=clip["ball"], poss=clip["poss"], order=order,
                        T=T, idx={o: j for j, o in enumerate(order)}, players=[], events=[])
    on = clip.get("on", np.ones((T, N), bool))
    vis = vis & on
    role = {o: ("GK" if r == "GK" else "CM") for o, r in zip(order, clip["role"])}
    anchors = {o: a for o, a in zip(order, clip["anchors"])} if clip.get("anchors") is not None else None
    rec = reconstruct(m, vis, mode, {}, on=on, role=role, anchors=anchors)
    pos, r80, p5 = rec.pos.copy(), rec.rad.copy(), rec.conf.copy()
    if model is not None:
        pos_obs = np.where(vis[..., None], clip["pos"], np.nan)
        X, idx = features(pos_obs, vis, on, clip["team"], clip["role"], clip["ball"], clip["poss"], clip["alive"], rec.pos,
                          mode, version=model.feat_version)
        if len(idx):
            seq = None
            if model.needs_context:
                C = hm.context_tensor(pos_obs, vis, on, clip["team"], clip["ball"], clip["poss"], clip["alive"], rec.pos)
                seq = hm.CtxStore([C], mode).windows(np.zeros(len(idx), np.int64), idx[:, 0], idx[:, 1])
            corr, sig = model.raw(X, seq)
            t, j = idx[:, 0], idx[:, 1]
            corr = np.where((clip["team"][j] == "B")[:, None], -corr, corr)
            pos[t, j] = np.clip(rec.pos[t, j] + corr, [0, 0], [L, W])
            r80[t, j], p5[t, j] = model.calibrate(sig, X)
    return pos, r80, p5


def model_path(method, mode):
    if method == "final_v3":  # existing production model (all 7 DFL matches, legacy ml.py run_final)
        return ROOT / "cv" / "weights" / f"hidden_{mode}_final_v3.pt"
    tag, fold = method.rsplit("_", 1)  # e.g. r13_mlp_final -> cv/weights/recon/r13_mlp/hidden_live_foldfinal.pt
    return ROOT / "cv" / "weights" / "recon" / tag / f"hidden_{mode}_fold{fold}.pt"


def gsr_clips(splits=("train", "valid")):
    for sp in splits:
        for path in sorted((GSR / sp).iterdir()):
            if (path / "Labels-GameState.json").exists():
                yield sp, load_clip(path)


def run(methods, clips=None):
    models = {(m, mode): hm.load_hidden(model_path(m, mode)) for m in methods if m != "base" for mode in ("live", "replay")}
    rows = []
    for sp, clip in (clips if clips is not None else gsr_clips()):
        if True:
            vis = np.isfinite(clip["pos"][..., 0]) & clip.get("on", True)
            ev = reentries(vis)
            if not ev:
                continue
            live_vis = vis.copy()
            for t, j, _ in ev:
                live_vis[t, j] = False
            rep_vis = vis.copy()
            rep_pts = []
            for t, j, g in ev:
                if t + EXT < len(vis) and vis[t:t + EXT + 1, j].all():
                    rep_vis[t:t + EXT, j] = False
                    rep_pts += [(t + k, j, g, k) for k in range(EXT)]
            preds = {}
            seg = {"pos": clip["pos"]}
            preds["live"] = {"last_position": (last_position(seg, live_vis), None, None),
                             "constant_velocity": (constant_velocity(seg, live_vis), None, None),
                             "formation": estimate(clip, live_vis, "live")}
            preds["replay"] = {"formation": estimate(clip, rep_vis, "replay")}
            for m in methods:
                if m == "base":
                    continue
                preds["live"][m] = estimate(clip, live_vis, "live", models[(m, "live")])
                preds["replay"][m] = estimate(clip, rep_vis, "replay", models[(m, "replay")])
            for mode, pts in (("live", [(t, j, g, 0) for t, j, g in ev]), ("replay", rep_pts)):
                vv = live_vis if mode == "live" else rep_vis
                for t, j, g, k in pts:
                    truth = clip["pos"][t, j]
                    mates = int((vv[t] & (clip["team"] == clip["team"][j])).sum())
                    r = {"split": sp, "game": clip["game"], "clip": clip["clip"], "mode": mode, "t": int(t), "j": int(j),
                         "gap_s": int(g) + k, "real_gap_s": int(g), "role": clip["role"][j], "visible_teammates": mates}
                    for name, (p, r80, p5) in preds[mode].items():
                        e = float(np.hypot(*(p[t, j] - truth))) if np.isfinite(p[t, j]).all() else np.nan
                        r[f"err:{name}"] = e
                        if r80 is not None:
                            r[f"r80:{name}"] = float(r80[t, j])
                            r[f"p5:{name}"] = float(p5[t, j])
                    rows.append(r)
    return rows


def summarize(rows):
    out = {}
    for mode in ("live", "replay"):
        R = [r for r in rows if r["mode"] == mode]
        if not R:
            continue
        names = [k[4:] for k in R[0] if k.startswith("err:")]
        games = sorted({r["game"] for r in R})
        res = {"n": len(R), "games": games, "clips": len({r["clip"] for r in R}), "methods": {}}
        for nm in names:
            e = np.array([r[f"err:{nm}"] for r in R])
            ok = np.isfinite(e)
            s = {"n": int(ok.sum()), "prediction_coverage": float(ok.mean()), "mean": float(e[ok].mean()),
                 "median": float(np.median(e[ok])), "rmse": float(np.sqrt((e[ok] ** 2).mean())),
                 "within_2m": float((e[ok] <= 2).mean()), "within_5m": float((e[ok] <= 5).mean()),
                 "within_10m": float((e[ok] <= 10).mean())}
            if f"r80:{nm}" in R[0]:
                rr = np.array([r[f"r80:{nm}"] for r in R])
                s["r80_coverage"] = float((e[ok] <= rr[ok]).mean())
                s["r80_mean_m"] = float(rr[ok].mean())
            s["per_game_mean"] = {g: float(np.nanmean([r[f"err:{nm}"] for r in R if r["game"] == g])) for g in games}
            s["game_mean"] = float(np.mean(list(s["per_game_mean"].values())))
            gaps = np.array([r["gap_s"] for r in R])
            s["by_gap"] = {f"{a}-{b}s": float(np.nanmean(e[(gaps >= a) & (gaps <= b)])) for a, b in ((1, 2), (3, 5), (6, 10), (11, 30))
                           if ((gaps >= a) & (gaps <= b)).sum() >= 30}
            s["by_gap_n"] = {f"{a}-{b}s": int(((gaps >= a) & (gaps <= b)).sum()) for a, b in ((1, 2), (3, 5), (6, 10), (11, 30))}
            res["methods"][nm] = s
        out[mode] = res
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="base,final_v3")
    ap.add_argument("--name", default="gsr")
    a = ap.parse_args()
    rows = run(a.methods.split(","))
    OUT.mkdir(parents=True, exist_ok=True)
    s = summarize(rows)
    (OUT / f"{a.name}_summary.json").write_text(json.dumps(s, indent=1))
    (OUT / f"{a.name}_rows.json").write_text(json.dumps(rows))
    for mode, res in s.items():
        print(f"\n{mode}: n={res['n']} clips={res['clips']} games={res['games']}")
        for nm, v in res["methods"].items():
            print(f"  {nm:28s} mean {v['mean']:.2f} median {v['median']:.2f} <=5m {v['within_5m']:.1%} game-mean {v['game_mean']:.2f}"
                  + (f" r80cov {v['r80_coverage']:.1%} r80 {v['r80_mean_m']:.1f}" if "r80_coverage" in v else "") + f" by_gap {v['by_gap']}")


def radius_scale_experiment(rows, method, mode="live"):
    """Broadcast-domain radius scale, 2-fold by GSR game group: fit s = 80th percentile of err / r80 on one group of
    games (SoccerNet train-split games), apply to the other (valid-split games), and swap. Reports coverage of the
    unscaled and scaled 80 % radius on the games NOT used to fit."""
    R = [r for r in rows if r["mode"] == mode and np.isfinite(r.get(f"err:{method}", np.nan)) and f"r80:{method}" in r]
    out = {}
    for fit, test in (("train", "valid"), ("valid", "train")):
        e_fit = np.array([r[f"err:{method}"] for r in R if r["split"] == fit])
        r_fit = np.array([r[f"r80:{method}"] for r in R if r["split"] == fit])
        e_te = np.array([r[f"err:{method}"] for r in R if r["split"] == test])
        r_te = np.array([r[f"r80:{method}"] for r in R if r["split"] == test])
        s = float(np.quantile(e_fit / r_fit, 0.8))
        out[f"fit_{fit}_test_{test}"] = {"scale": round(s, 3), "n_test": int(len(e_te)),
                                         "coverage_unscaled": round(float((e_te <= r_te).mean()), 3),
                                         "coverage_scaled": round(float((e_te <= s * r_te).mean()), 3),
                                         "mean_radius_unscaled_m": round(float(r_te.mean()), 2),
                                         "mean_radius_scaled_m": round(float(s * r_te.mean()), 2)}
    return out
