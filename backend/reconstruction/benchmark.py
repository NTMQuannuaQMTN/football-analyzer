"""Unified held-out benchmark for hidden-player reconstruction on real DFL tracking (7 matches, 14 halves).

Every method is scored on the same outer folds, masks, mask seeds and hidden player-seconds:
  fold i      test match = dfl_matches()[i]; learned methods use checkpoints whose training excluded it
  masks       sc_camera x2 instances (real SkillCorner broadcast footprints + real detection misses, moved onto the
              DFL ball), k=14 / k=10 / k=6 (nearest-to-camera-centre; simulated), seeds disjoint from training
  scored      every on-pitch player-second that the mask hides. Visible players are passed through exactly (they are
              GROUND TRUTH here, not detections), so they are counted but never scored as predictions.

Per hidden row it stores the error plus context (time since / until sighting, why hidden, visible team-mates, role,
phase, transitions, restarts, calibrated radius and P(err <= 5 m)). `report` aggregates per match and overall.

    .venv/bin/python -m backend.reconstruction.benchmark run --run r13 --methods base,k7r_v3,r13_mlp,r13_gru
    .venv/bin/python -m backend.reconstruction.benchmark report --run r13
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from backend.reconstruction import hidden_models as hm
from backend.reconstruction.datasets import CACHE, L, ROLES, W, dfl_matches, parse_dfl, point_in_quad
from backend.reconstruction.ml import apply_calibration, features
from backend.reconstruction.postprocess import limit_speed
from backend.reconstruction.real_benchmark import constant_velocity, formation, last_position, make_mask, mask_sc

ROOT = Path(__file__).resolve().parents[2]
OUT = CACHE.parent / "recon_benchmark"
EVAL_SEED = 777000
EVAL_INSTANCES = [("sc_camera", 0), ("sc_camera", 1), ("k=14", 0), ("k=10", 0), ("k=6", 0)]
CAT = {"observed": 0, "outside_view": 1, "in_view_missed": 2, "camera_cut": 3}
PHASE = {"in_possession": 0, "out_of_possession": 1, "dead_ball": 2, "unknown": 3}
SINCE_BINS = [(0, 2), (3, 5), (6, 10), (11, 30), (31, 60), (61, 120), (121, 10 ** 6)]
VIS_BINS = [(0, 0), (1, 2), (3, 5), (6, 11)]


def checkpoint(method, mode, fold):
    """Path of the held-out checkpoint for a learned method, or None."""
    if method == "k7r_v3":  # existing 7-fold v3 MLP (refit), produced by ml.py --k7 --refit --feat 3
        return ROOT / "cv" / "weights" / f"hidden_{mode}_k7r_{fold}_v3.pt"
    return ROOT / "cv" / "weights" / "recon" / method / f"hidden_{mode}_fold{fold}.pt"


def _transitions(poss, alive, T, win=5):
    """near_transition: within win s of a change of the team in possession; near_restart: within win s of play
    resuming after a dead ball. Uses the recorded possession / ball status (evaluation context only)."""
    last, change = -1, []
    for t in range(T):
        if poss[t] >= 0:
            if last >= 0 and poss[t] != last:
                change.append(t)
            last = poss[t]
    restart = [t for t in range(1, T) if alive[t] and not alive[t - 1]]

    def near(ev):
        m = np.zeros(T, bool)
        for t in ev:
            m[max(0, t - win):t + win + 1] = True
        return m
    return near(change), near(restart)


def instance_context(seg, vis, quads):
    """Row metadata for every on-pitch hidden player-second."""
    T, N = vis.shape
    on = ~np.isnan(seg["pos"][..., 0])
    from backend.reconstruction.ml import since_until
    since, until = since_until(vis)
    team = seg["team"]
    cat = np.full((T, N), CAT["outside_view"], np.int8)
    if quads is not None:
        for t in range(T):
            if np.isnan(quads[t]).any():
                cat[t] = CAT["camera_cut"]
            else:
                inside = point_in_quad(np.nan_to_num(seg["pos"][t], nan=-999), quads[t])
                cat[t, inside] = CAT["in_view_missed"]
    cat[vis] = CAT["observed"]
    nmates = np.zeros((T, N), np.int8)
    for tm in ("A", "B"):
        c = (vis & (team == tm)[None]).sum(1)
        nmates[:, team == tm] = c[:, None]
    tidx = np.where(team == "A", 0, 1)
    phase = np.where(~seg["alive"][:, None], PHASE["dead_ball"],
                     np.where(seg["poss"][:, None] < 0, PHASE["unknown"],
                              np.where(seg["poss"][:, None] == tidx[None], PHASE["in_possession"], PHASE["out_of_possession"])))
    ntr, nrs = _transitions(seg["poss"], seg["alive"], T)
    role = np.array([ROLES.index(r) if r in ROLES else -1 for r in seg["role"]])
    hid = on & ~vis
    t, j = np.where(hid)
    s = since[t, j]
    u = until[t, j]
    return {"t": t.astype(np.int32), "j": j.astype(np.int16),
            "since": np.where(np.isfinite(s), s, -1).astype(np.int32),
            "gap": np.where(np.isfinite(s) & np.isfinite(u), s + u - 1, -1).astype(np.int32),
            "cat": cat[t, j], "nmates": nmates[t, j] - 0, "role": role[j].astype(np.int8),
            "team": (team[j] == "B").astype(np.int8), "phase": phase[t, j].astype(np.int8),
            "near_transition": ntr[t], "near_restart": nrs[t]}, hid, int((vis & on).sum())


def check_held_out(ck, mid, name="checkpoint"):
    """A checkpoint may score a match only if it records that match as its test match and never trained on it."""
    if mid in (ck.get("train") or []):
        raise RuntimeError(f"{name} trained on {mid}")
    if ck.get("test") != mid:
        raise RuntimeError(f"{name} does not record {mid} as its held-out test match (test={ck.get('test')})")


def _gk_clamp(pos, src_inf, seg):
    for j, r in enumerate(seg["role"]):
        if r == "GK":
            rows = src_inf[:, j]
            if seg["team"][j] == "A":
                pos[rows, j, 0] = np.minimum(pos[rows, j, 0], 25)
            else:
                pos[rows, j, 0] = np.maximum(pos[rows, j, 0], L - 25)
    return pos


def run_fold(fold, methods, run, speed_limit=True):
    ms = dfl_matches()
    mid = ms[fold]
    info, segs = parse_dfl(mid)
    learned = [m for m in methods if m != "base"]
    models = {}
    for m in learned:
        for mode in ("live", "replay"):
            p = checkpoint(m, mode, fold)
            if not p.exists():
                raise FileNotFoundError(p)
            mdl = hm.load_hidden(p)
            check_held_out(mdl.ck, mid, p)
            models[(m, mode)] = mdl
    rows = {"meta": [], "err": {}, "r80": {}, "p5": {}}
    counts = []
    for si, seg in enumerate(segs):
        on = ~np.isnan(seg["pos"][..., 0])
        for k, (mk, rep) in enumerate(EVAL_INSTANCES):
            seed = EVAL_SEED + 1000 * fold + 100 * si + 10 * k + rep
            if mk == "sc_camera":
                vis, quads = mask_sc(seg, seed, return_quads=True)
            else:
                vis, quads = make_mask(seg, mk, seed), None
            vis = vis & on
            meta, hid, n_obs = instance_context(seg, vis, quads)
            n = len(meta["t"])
            meta.update({"half": np.full(n, si, np.int8), "inst": np.full(n, k, np.int8)})
            rows["meta"].append(meta)
            counts.append({"half": si, "mask": mk, "rep": rep, "hidden": n, "observed": n_obs,
                           "on_pitch": int(on.sum()), "seconds": int(len(vis))})
            t, j = meta["t"], meta["j"]

            def put(name, pos, r80=None, p5=None):
                e = np.linalg.norm(pos[t, j] - seg["pos"][t, j], axis=1)
                rows["err"].setdefault(name, []).append(e.astype(np.float32))
                rows["r80"].setdefault(name, []).append((r80[t, j] if r80 is not None else np.full(n, np.nan)).astype(np.float32))
                rows["p5"].setdefault(name, []).append((p5[t, j] if p5 is not None else np.full(n, np.nan)).astype(np.float32))

            if "base" in methods:
                put("last_position", last_position(seg, vis))
                put("constant_velocity", constant_velocity(seg, vis))
            pos_obs = np.where(vis[..., None], seg["pos"], np.nan)
            for mode in ("live", "replay"):
                rec = formation(info, seg, vis, mode)
                if "base" in methods:
                    put(f"formation_{mode}", rec.pos, rec.rad, rec.conf)
                if not learned:
                    continue
                X, idx = features(pos_obs, vis, on, seg["team"], seg["role"], seg["ball"], seg["poss"], seg["alive"],
                                  rec.pos, mode, version=3)
                store = None
                for m in learned:
                    mdl = models[(m, mode)]
                    seq = None
                    if mdl.needs_context:
                        if store is None:
                            C = hm.context_tensor(pos_obs, vis, on, seg["team"], seg["ball"], seg["poss"], seg["alive"], rec.pos)
                            store = hm.CtxStore([C], mode)
                        seq = store.windows(np.zeros(len(idx), np.int64), idx[:, 0], idx[:, 1])
                    corr, sig = mdl.raw(X, seq)
                    ti, ji = idx[:, 0], idx[:, 1]
                    corr = np.where((seg["team"][ji] == "B")[:, None], -corr, corr)
                    pos = rec.pos.copy()
                    pos[ti, ji] = np.clip(rec.pos[ti, ji] + corr, [0, 0], [L, W])
                    r80 = np.full(vis.shape, np.nan)
                    p5 = np.full(vis.shape, np.nan)
                    r80[ti, ji], p5[ti, ji] = mdl.calibrate(sig, X)
                    src_inf = on & ~vis
                    pos = _gk_clamp(pos, src_inf, seg)
                    put(f"{m}_{mode}", pos, r80, p5)
                    if speed_limit:
                        put(f"{m}_{mode}+speed", limit_speed(pos, vis, on, mode == "live"), r80, p5)
    out = {"match": mid, "fold": fold, "counts": counts,
           "meta": {k: np.concatenate([m[k] for m in rows["meta"]]) for k in rows["meta"][0]}}
    for key in ("err", "r80", "p5"):
        out[key] = {name: np.concatenate(v) for name, v in rows[key].items()}
    d = OUT / run
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"fold{fold}.npz", d=np.array(out, dtype=object))
    return out


# ----------------------------------------------------------------------------- reporting
def load_run(run, folds=range(7)):
    res = []
    for f in folds:
        p = OUT / run / f"fold{f}.npz"
        if p.exists():
            res.append(np.load(p, allow_pickle=True)["d"].item())
    return res


def merge_runs(runs, folds=range(7)):
    """Combine methods evaluated in different runs (rows are identical: same masks, seeds and order)."""
    merged = None
    for run in runs:
        for f in load_run(run, folds):
            if merged is None:
                merged = {}
            cur = merged.setdefault(f["fold"], f)
            if cur is not f:
                assert len(cur["meta"]["t"]) == len(f["meta"]["t"]) and (cur["meta"]["t"] == f["meta"]["t"]).all()
                for key in ("err", "r80", "p5"):
                    cur[key].update(f[key])
    return [merged[k] for k in sorted(merged)] if merged else []


def stats(e, r80=None, p5=None):
    e = e[np.isfinite(e)] if r80 is None else e
    if not len(e):
        return None
    s = {"n": int(len(e)), "mean": float(e.mean()), "median": float(np.median(e)), "rmse": float(np.sqrt((e ** 2).mean())),
         "p90": float(np.percentile(e, 90)), "within_2m": float((e <= 2).mean()), "within_5m": float((e <= 5).mean()),
         "within_10m": float((e <= 10).mean())}
    if r80 is not None and np.isfinite(r80).any():
        ok = np.isfinite(r80)
        s["r80_coverage"] = float((e[ok] <= r80[ok]).mean())
        s["r80_mean_m"] = float(r80[ok].mean())
    if p5 is not None and np.isfinite(p5).any():
        ok = np.isfinite(p5)
        s["p5_predicted"] = float(p5[ok].mean())
        s["p5_observed"] = float((e[ok] <= 5).mean())
    return s


def _group_masks(meta):
    g = {}
    for a, b in SINCE_BINS:
        g[f"since_{a}-{b if b < 10 ** 6 else 'inf'}s"] = (meta["since"] >= a) & (meta["since"] <= b)
    g["since_never_seen"] = meta["since"] < 0
    for a, b in VIS_BINS:
        g[f"visible_teammates_{a}-{b}"] = (meta["nmates"] >= a) & (meta["nmates"] <= b)
    for name, c in CAT.items():
        if c:
            g[f"why_{name}"] = meta["cat"] == c
    g["long_unseen_gt30s"] = meta["since"] > 30
    for name, c in PHASE.items():
        g[f"phase_{name}"] = meta["phase"] == c
    g["near_possession_change_5s"] = meta["near_transition"]
    g["near_restart_5s"] = meta["near_restart"]
    for i, r in enumerate(ROLES):
        g[f"role_{r}"] = meta["role"] == i
    g["team_home"] = meta["team"] == 0
    g["team_away"] = meta["team"] == 1
    return g


def method_table(folds, method, inst_filter):
    """Overall (sample-weighted), match-mean, per-match and per-group stats for one method."""
    per_match, allE, allR, allP, metas = {}, [], [], [], []
    for f in folds:
        if method not in f["err"]:
            return None
        sel = np.isin(f["meta"]["inst"], inst_filter)
        e, r, p = f["err"][method][sel], f["r80"][method][sel], f["p5"][method][sel]
        fin = np.isfinite(e)
        st = stats(e[fin], r[fin], p[fin])
        st["prediction_coverage"] = float(fin.mean())
        per_match[f["match"]] = st
        allE.append(e), allR.append(r), allP.append(p)
        metas.append({k: v[sel] for k, v in f["meta"].items()})
    E, R, P = np.concatenate(allE), np.concatenate(allR), np.concatenate(allP)
    meta = {k: np.concatenate([m[k] for m in metas]) for k in metas[0]}
    fin = np.isfinite(E)
    overall = stats(E[fin], R[fin], P[fin])
    overall["prediction_coverage"] = float(fin.mean())
    mm = {k: float(np.mean([v[k] for v in per_match.values()])) for k in ("mean", "median", "rmse", "within_5m")}
    mm["mean_sd_across_matches"] = float(np.std([v["mean"] for v in per_match.values()]))
    groups = {}
    for g, m in _group_masks(meta).items():
        m = m & fin
        if m.sum() >= 200:
            groups[g] = stats(E[m], R[m], P[m])
    rel = None
    if np.isfinite(P).any():
        ok = fin & np.isfinite(P)
        edges = np.linspace(0, 1, 6)
        rel = []
        for a, b in zip(edges[:-1], edges[1:]):
            s = ok & (P >= a) & (P < b + (1e-9 if b == 1 else 0))
            if s.sum() >= 100:
                rel.append({"p5_bin": f"{a:.1f}-{b:.1f}", "n": int(s.sum()), "predicted": float(P[s].mean()),
                            "observed": float((E[s] <= 5).mean())})
    return {"overall": overall, "match_mean": mm, "per_match": per_match, "groups": groups, "p5_reliability": rel}


def report(runs, out_name, folds=range(7)):
    fl = merge_runs(runs, folds)
    if not fl:
        raise SystemExit("no folds found")
    methods = sorted(fl[0]["err"])
    res = {"runs": runs, "folds": [f["fold"] for f in fl], "matches": [f["match"] for f in fl],
           "counts": {f["match"]: f["counts"] for f in fl}, "masks": {}}
    for name, insts in {"sc_camera": [0, 1], "k=14": [2], "k=10": [3], "k=6": [4]}.items():
        res["masks"][name] = {m: method_table(fl, m, insts) for m in methods}
    # paired per-match comparison against the formation estimator and the existing model (sc_camera)
    pairs = {}
    for m in methods:
        for ref in ("formation_live", "formation_replay", "k7r_v3_live", "k7r_v3_replay", "k7r_v3_live+speed", "k7r_v3_replay+speed"):
            if ref == m or ref not in methods or m.split("_")[-1].split("+")[0] != ref.split("_")[-1].split("+")[0]:
                continue
            a = res["masks"]["sc_camera"][m]["per_match"]
            b = res["masks"]["sc_camera"][ref]["per_match"]
            d = [a[k]["mean"] - b[k]["mean"] for k in a]
            pairs[f"{m} vs {ref}"] = {"mean_diff_m": float(np.mean(d)), "sd": float(np.std(d)), "wins": int(sum(x < 0 for x in d)),
                                      "matches": len(d), "per_match": dict(zip(a, [round(x, 3) for x in d]))}
    res["paired_sc_camera"] = pairs
    out = OUT / f"report_{out_name}.json"
    out.write_text(json.dumps(res, indent=1, default=float))
    return res


def markdown(res, methods=None):
    lines = []
    methods = methods or list(res["masks"]["sc_camera"])
    lines.append("| Method | n | Mean | Median | RMSE | P90 | <=2 m | <=5 m | <=10 m | Match-mean (sd) | r80 cover | r80 mean |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in methods:
        t = res["masks"]["sc_camera"].get(m)
        if not t:
            continue
        o, mm = t["overall"], t["match_mean"]
        lines.append(f"| {m} | {o['n']:,} | {o['mean']:.2f} | {o['median']:.2f} | {o['rmse']:.2f} | {o['p90']:.2f} | "
                     f"{o['within_2m']:.1%} | {o['within_5m']:.1%} | {o['within_10m']:.1%} | {mm['mean']:.2f} ({mm['mean_sd_across_matches']:.2f}) | "
                     f"{o.get('r80_coverage', float('nan')):.1%} | {o.get('r80_mean_m', float('nan')):.1f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--run", required=True)
    r.add_argument("--methods", default="base,k7r_v3")
    r.add_argument("--folds", default="0,1,2,3,4,5,6")
    p = sub.add_parser("report")
    p.add_argument("--runs", required=True, help="comma-separated runs to merge")
    p.add_argument("--name", required=True)
    a = ap.parse_args()
    if a.cmd == "run":
        for f in [int(x) for x in a.folds.split(",")]:
            t0 = time.time()
            run_fold(f, a.methods.split(","), a.run)
            print(f"fold {f} done in {time.time() - t0:.0f}s", flush=True)
    else:
        res = report(a.runs.split(","), a.name)
        print(markdown(res))


# ----------------------------------------------------------------------------- broadcast-context degradation
def degraded_eval(fold, methods, which="validation", window=30, seed=EVAL_SEED + 55):
    """DFL scored under GSR-like broadcast context: independent `window`-s clips, no possession / ball-status feed
    (poss=-1, alive=True), no positional roles or DFL anchors (GK vs outfield only), real-footprint mask.
    which="validation": the time-blocked tail (t >= 2200 s) of the fold's TRAINING matches -> may select models.
    which="test": the fold's held-out match -> report only. Scored: hidden player-seconds after a sighting in the clip."""
    from types import SimpleNamespace
    from backend.reconstruction.estimator import reconstruct
    from backend.reconstruction.train_hidden import VAL_FROM_T, fold_split
    train_ms, test = fold_split(fold)
    mids = train_ms if which == "validation" else [test]
    models = {(m, mode): hm.load_hidden(checkpoint(m, mode, fold)) for m in methods for mode in ("live", "replay")}
    for (m, mode), mdl in models.items():
        if which == "test":
            check_held_out(mdl.ck, test, f"{m}/{mode}")
    errs = {}
    for mid in mids:
        _, segs = parse_dfl(mid)
        for si, seg in enumerate(segs):
            on_all = ~np.isnan(seg["pos"][..., 0])
            vis_all = mask_sc(seg, seed + 1000 * fold + 10 * si) & on_all
            t0s = range(VAL_FROM_T if which == "validation" else 0, len(vis_all) - window + 1, window)
            for t0 in t0s:
                sl = slice(t0, t0 + window)
                pos, vis, on = seg["pos"][sl], vis_all[sl], on_all[sl]
                team, T, N = seg["team"], window, len(seg["team"])
                role = np.where(seg["role"] == "GK", "GK", "CM")
                ball = seg["ball"][sl]
                poss, alive = np.full(T, -1), np.ones(T, bool)
                order = [f"{tm}:{j}" for j, tm in enumerate(team)]
                mm = SimpleNamespace(xy=np.where(vis[..., None], pos, np.nan), ball=ball, poss=poss, order=order, T=T,
                                     idx={o: j for j, o in enumerate(order)}, players=[], events=[])
                from backend.reconstruction.ml import since_until
                since, _ = since_until(vis)
                hid = on & ~vis & np.isfinite(since)
                if not hid.any():
                    continue
                for mode in ("live", "replay"):
                    rec = reconstruct(mm, vis, mode, {}, on=on, role={o: r for o, r in zip(order, role)}, anchors=None)
                    e = np.linalg.norm(rec.pos - pos, axis=2)[hid]
                    errs.setdefault(f"formation_{mode}", []).append(e)
                    for m in methods:
                        idx, corr, sig = hm.predict_segment(models[(m, mode)], pos, vis, on, team, role, ball, poss, alive,
                                                            rec.pos, mode)
                        p = rec.pos.copy()
                        if len(idx):
                            p[idx[:, 0], idx[:, 1]] = np.clip(rec.pos[idx[:, 0], idx[:, 1]] + corr, [0, 0], [L, W])
                        errs.setdefault(f"{m}_{mode}", []).append(np.linalg.norm(p - pos, axis=2)[hid])
    return {k: {"n": int(sum(len(x) for x in v)), "mean": float(np.concatenate(v).mean())} for k, v in errs.items()}


def degraded_run(methods, which, folds=range(7), name=None):
    out = {}
    for f in folds:
        out[f] = degraded_eval(f, methods, which)
        print(f, {k: round(v["mean"], 2) for k, v in out[f].items()}, flush=True)
    agg = {}
    for k in out[next(iter(out))]:
        agg[k] = {"fold_mean": float(np.mean([out[f][k]["mean"] for f in out])), "n": int(sum(out[f][k]["n"] for f in out))}
    res = {"which": which, "folds": out, "aggregate": agg}
    (OUT / f"degraded_{which}_{name or '_'.join(methods)}.json").write_text(json.dumps(res, indent=1, default=float))
    return res


def full_validation(fold, methods, seed=EVAL_SEED + 77):
    """Full-context score on the time-blocked tail (t >= 2200 s) of the fold's TRAINING matches: whole-half history,
    possession / ball status, DFL roles and anchors, real-footprint mask. Same rows for every method -> selection."""
    from backend.reconstruction.train_hidden import VAL_FROM_T, fold_split
    train_ms, _ = fold_split(fold)
    models = {(m, mode): hm.load_hidden(checkpoint(m, mode, fold)) for m in methods for mode in ("live", "replay")}
    errs = {}
    for mid in train_ms:
        info, segs = parse_dfl(mid)
        for si, seg in enumerate(segs):
            on = ~np.isnan(seg["pos"][..., 0])
            vis = mask_sc(seg, seed + 1000 * fold + 10 * si) & on
            hid = on & ~vis
            hid[:VAL_FROM_T] = False
            pos_obs = np.where(vis[..., None], seg["pos"], np.nan)
            for mode in ("live", "replay"):
                rec = formation(info, seg, vis, mode)
                for m in methods:
                    idx, corr, sig = hm.predict_segment(models[(m, mode)], seg["pos"], vis, on, seg["team"], seg["role"],
                                                        seg["ball"], seg["poss"], seg["alive"], rec.pos, mode,
                                                        t_index=range(VAL_FROM_T, len(vis)))
                    p = rec.pos.copy()
                    if len(idx):
                        p[idx[:, 0], idx[:, 1]] = np.clip(rec.pos[idx[:, 0], idx[:, 1]] + corr, [0, 0], [L, W])
                    p = _gk_clamp(p, on & ~vis, seg)
                    errs.setdefault(f"{m}_{mode}", []).append(np.linalg.norm(p - seg["pos"], axis=2)[hid])
    del pos_obs
    return {k: {"n": int(sum(len(x) for x in v)), "mean": float(np.concatenate(v).mean())} for k, v in errs.items()}


def full_validation_run(methods, folds=range(7), name=None):
    out = {f: full_validation(f, methods) for f in folds}
    for f in folds:
        print(f, {k: round(v["mean"], 2) for k, v in out[f].items()}, flush=True)
    agg = {k: {"fold_mean": float(np.mean([out[f][k]["mean"] for f in out]))} for k in out[next(iter(out))]}
    res = {"which": "validation_full", "folds": out, "aggregate": agg}
    (OUT / f"fullval_{name or '_'.join(methods)}.json").write_text(json.dumps(res, indent=1, default=float))
    return res


# ----------------------------------------------------------------------------- identity errors (tracker swaps)
def identity_swap(seg, vis, rate, seed):
    """Simulated tracker identity errors: each appearance (run of visible seconds) of a player is, with probability
    `rate`, recorded under another on-pitch player of the same team who is not visible then. Returns (obs_pos, vis')
    as an identity-blind pipeline would see them; truth is untouched."""
    rng = np.random.default_rng(seed)
    T, N = vis.shape
    on = ~np.isnan(seg["pos"][..., 0])
    obs = np.where(vis[..., None], seg["pos"], np.nan)
    v2 = vis.copy()
    team = seg["team"]
    for j in range(N):
        t = 0
        while t < T:
            if not vis[t, j]:
                t += 1
                continue
            t1 = t
            while t1 < T and vis[t1, j]:
                t1 += 1
            if rng.random() < rate:
                cands = [k for k in range(N) if k != j and team[k] == team[j] and on[t:t1, k].all() and not v2[t:t1, k].any()]
                if cands:
                    k = cands[rng.integers(len(cands))]
                    v2[t:t1, k], obs[t:t1, k] = True, seg["pos"][t:t1, j]
                    v2[t:t1, j], obs[t:t1, j] = False, np.nan
            t = t1
    return obs, v2


def identity_eval(methods=("r13_mlp_aug_ens3",), rates=(0.0, 0.1, 0.3, 0.65), folds=range(7)):
    """Held-out DFL, real-footprint mask, replay + live. Scores every on-pitch player-second against the player's TRUE
    position, split into: shown as observed under the right id, shown as observed but actually another player
    (wrong identity), and estimated."""
    res = {}
    for f in folds:
        mid = dfl_matches()[f]
        info, segs = parse_dfl(mid)
        models = {(m, mode): hm.load_hidden(checkpoint(m, mode, f)) for m in methods for mode in ("live", "replay")}
        for (m, mode), mdl in models.items():
            check_held_out(mdl.ck, mid, m)
        for si, seg in enumerate(segs):
            on = ~np.isnan(seg["pos"][..., 0])
            vis = mask_sc(seg, EVAL_SEED + 1000 * f + 100 * si) & on
            for rate in rates:
                obs, v2 = identity_swap(seg, vis, rate, EVAL_SEED + 7 * f + si)
                seg2 = dict(seg, pos=np.where(v2[..., None], obs, seg["pos"]))  # pipeline input: identity-blind
                wrong = v2 & ~vis | (v2 & vis & (np.linalg.norm(np.nan_to_num(obs) - np.nan_to_num(seg["pos"]), axis=2) > 1e-6))
                for mode in ("live", "replay"):
                    rec = formation(info, seg2, v2, mode)
                    for m in methods:
                        idx, corr, sig = hm.predict_segment(models[(m, mode)], seg2["pos"], v2, on, seg["team"], seg["role"],
                                                            seg["ball"], seg["poss"], seg["alive"], rec.pos, mode)
                        p = rec.pos.copy()
                        if len(idx):
                            p[idx[:, 0], idx[:, 1]] = np.clip(rec.pos[idx[:, 0], idx[:, 1]] + corr, [0, 0], [L, W])
                        p = np.where(v2[..., None], obs, p)  # observed rows: shown where the (possibly wrong) detection is
                        p = limit_speed(_gk_clamp(p, on & ~v2, seg), v2, on, mode == "live")
                        e = np.linalg.norm(p - seg["pos"], axis=2)
                        d = res.setdefault((m, mode, rate), {"est": [], "wrong": [], "right": [], "all": []})
                        d["est"].append(e[on & ~v2])
                        d["wrong"].append(e[wrong & on])
                        d["right"].append(e[v2 & ~wrong & on])
                        d["all"].append(e[on])
        print("fold", f, "done", flush=True)
    out = {}
    for (m, mode, rate), d in res.items():
        c = {k: np.concatenate(v) for k, v in d.items()}
        out[f"{m}|{mode}|{rate}"] = {k: {"n": int(len(v)), "mean": round(float(v.mean()), 2) if len(v) else None} for k, v in c.items()}
    (OUT / "identity_swaps.json").write_text(json.dumps(out, indent=1))
    return out
