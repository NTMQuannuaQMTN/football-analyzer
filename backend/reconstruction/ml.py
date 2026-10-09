"""Learned hidden-player estimator (trained on DFL ground truth, camera-masked).

Model: MLP over structured state features that predicts a CORRECTION to the team-shape (formation)
estimator plus a per-prediction uncertainty. Relationships between players enter through pooled
relational features (nearest visible teammates / opponents, team centroids, ball, possession).
  LIVE model    features from t-N..t only (prior = live formation estimate)
  REPLAY model  also next sighting (position, time until) + replay formation prior
Uncertainty: head predicts log sigma (Gaussian NLL, mean detached); then calibrated EMPIRICALLY on the
validation match: sigma decile -> 80% error radius and P(error <= 5 m).

    .venv/bin/python -m backend.reconstruction.ml --fold 0     # build features, train live+replay, evaluate
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from backend.reconstruction.datasets import CACHE, L, ROLES, W, parse_dfl
from backend.reconstruction.real_benchmark import (constant_velocity, errors, formation, last_position, make_mask,
                                                   split, summarize, breakdown)

WEIGHTS = Path(__file__).resolve().parents[2] / "cv" / "weights"
TRAIN_MASKS = ["sc_camera", "sc_camera", "k=16", "k=12", "k=8", "k=6", "rand=10"]
EVAL_MASKS = ["sc_camera", "k=18", "k=16", "k=14", "k=12", "k=10", "k=8", "k=6", "rand=10"]
K_NEI = 3
FEAT_VERSION = 1  # 2 = + temporal / team-relative features (slot memory, team extent, offside line, ball velocity)


def _own(team_is_b, p):
    """Own-team frame (each team attacks +x). p (..., 2)."""
    q = np.array(p, dtype=float, copy=True)
    if team_is_b:
        q[..., 0] = L - q[..., 0]
        q[..., 1] = W - q[..., 1]
    return q


def since_until(vis):
    T, N = vis.shape
    since = np.full((T, N), np.inf)
    until = np.full((T, N), np.inf)
    last = np.full(N, -np.inf)
    for t in range(T):
        last = np.where(vis[t], t, last)
        since[t] = t - last
    nxt = np.full(N, np.inf)
    for t in range(T - 1, -1, -1):
        nxt = np.where(vis[t], t, nxt)
        until[t] = nxt - t
    return since, until


def features(pos_obs, vis, on, team, role, ball, poss, alive, prior, mode, t_index=None, version=None):
    """Feature rows for every hidden on-pitch (t, j). pos_obs = positions where visible (others ignored).
    Returns X (n, F), idx (n, 2) [t, j]. Uses only t' <= t in live mode."""
    version = FEAT_VERSION if version is None else version
    T, N = vis.shape
    since, until = since_until(vis)
    if version >= 3:
        dead_for = np.zeros(T)
        run = 0
        for t in range(T):
            run = 0 if (alive is None or alive[t]) else run + 1
            dead_for[t] = run
    if version >= 2:
        # visible-team centroid per second (world frame), slot offsets at the last / next sighting, smoothed velocity
        cen = {}
        for tm in ("A", "B"):
            msk = (team == tm)[None, :] & vis
            cnt = msk.sum(1)
            sm = np.where(msk[..., None], np.nan_to_num(pos_obs), 0).sum(1)
            c = np.where(cnt[:, None] > 0, sm / np.maximum(cnt, 1)[:, None], np.nan)
            cen[tm] = c
        cen_j = np.stack([cen[tm] for tm in team], 1)  # (T, N, 2)
        off_last = np.full((T, N, 2), np.nan)
        vel3 = np.zeros((T, N, 2))
        cur_off = np.full((N, 2), np.nan)
        hist = [[] for _ in range(N)]
        cur_v = np.zeros((N, 2))
        for t in range(T):
            for j in np.where(vis[t])[0]:
                if np.isfinite(cen_j[t, j]).all():
                    cur_off[j] = pos_obs[t, j] - cen_j[t, j]
                hist[j].append((t, pos_obs[t, j]))
                hist[j] = hist[j][-4:]
                h = hist[j]
                if len(h) >= 2 and h[-1][0] - h[0][0] <= 6:
                    cur_v[j] = (h[-1][1] - h[0][1]) / max(1, h[-1][0] - h[0][0])
            off_last[t], vel3[t] = cur_off, cur_v
        off_next = np.full((T, N, 2), np.nan)
        if mode == "replay":
            c = np.full((N, 2), np.nan)
            for t in range(T - 1, -1, -1):
                for j in np.where(vis[t])[0]:
                    if np.isfinite(cen_j[t, j]).all():
                        c[j] = pos_obs[t, j] - cen_j[t, j]
                off_next[t] = c
    last = np.full((T, N, 2), np.nan)
    lastb = np.full((T, N, 2), np.nan)
    vel = np.zeros((T, N, 2))
    nxt = np.full((T, N, 2), np.nan)
    cur = np.full((N, 2), np.nan)
    curb = np.full((N, 2), np.nan)
    cv = np.zeros((N, 2))
    for t in range(T):
        seen = vis[t]
        if t > 0:
            cont = seen & vis[t - 1]
            cv[cont] = pos_obs[t, cont] - pos_obs[t - 1, cont]
            cv[seen & ~vis[t - 1]] = 0
        cur[seen] = pos_obs[t, seen]
        curb[seen] = ball[t]
        last[t], lastb[t], vel[t] = cur, curb, cv
    if mode == "replay":
        c = np.full((N, 2), np.nan)
        for t in range(T - 1, -1, -1):
            c[vis[t]] = pos_obs[t, vis[t]]
            nxt[t] = c
    is_b = (team == "B")
    role_oh = np.array([[r == k for k in ROLES] for r in role], dtype=np.float32)
    rows, idx = [], []
    tt = range(T) if t_index is None else t_index
    for t in tt:
        hid = np.where(on[t] & ~vis[t])[0]
        if not len(hid):
            continue
        visible = np.where(vis[t])[0]
        n_cut = float(len(visible) == 0)
        for j in hid:
            b = bool(is_b[j])
            o = lambda p: _own(b, p)
            pr = o(prior[t, j])
            bl = o(ball[t]) if np.isfinite(ball[t]).all() else np.array([L / 2, W / 2])
            mates = [k for k in visible if team[k] == team[j]]
            opps = [k for k in visible if team[k] != team[j]]
            mp = o(pos_obs[t, mates]) if mates else np.zeros((0, 2))
            op = o(pos_obs[t, opps]) if opps else np.zeros((0, 2))
            mc = mp.mean(0) if len(mp) else bl
            oc = op.mean(0) if len(op) else bl
            nei = []
            for P in (mp, op):
                d = np.linalg.norm(P - pr, axis=1) if len(P) else np.zeros(0)
                order = np.argsort(d)[:K_NEI]
                for i in range(K_NEI):
                    if i < len(order):
                        r = (P[order[i]] - pr) / 20
                        nei += [r[0], r[1], 1.0]
                    else:
                        nei += [0.0, 0.0, 0.0]
            seen_before = np.isfinite(since[t, j])
            ls = o(last[t, j]) if seen_before else pr
            lb = o(lastb[t, j]) if seen_before and np.isfinite(lastb[t, j]).all() else bl
            v = (vel[t, j] * (-1 if b else 1)) if seen_before else np.zeros(2)
            ps = 0.0 if poss[t] < 0 else (1.0 if poss[t] == (1 if b else 0) else -1.0)
            f = [pr[0] / L, pr[1] / W, ls[0] / L, ls[1] / W, np.log1p(min(since[t, j], 600)) / 4 if seen_before else 1.6,
                 float(not seen_before), v[0] / 8, v[1] / 8, bl[0] / L, bl[1] / W, lb[0] / L, lb[1] / W,
                 (ls[0] - pr[0]) / 20, (ls[1] - pr[1]) / 20,
                 len(mates) / 11, len(opps) / 11, mc[0] / L, mc[1] / W, oc[0] / L, oc[1] / W, ps,
                 float(alive[t]) if alive is not None else 1.0, n_cut]
            if mode == "replay":
                has = np.isfinite(until[t, j])
                nx = o(nxt[t, j]) if has else pr
                f += [nx[0] / L, nx[1] / W, (nx[0] - pr[0]) / 20, (nx[1] - pr[1]) / 20,
                      np.log1p(min(until[t, j], 600)) / 4 if has else 1.6, float(has)]
            if version >= 2:
                sgn = -1.0 if b else 1.0
                bv = (ball[t] - ball[t - 1]) * sgn if t > 0 and np.isfinite(ball[t]).all() and np.isfinite(ball[t - 1]).all() else np.zeros(2)
                ext = [(mp[:, 0].max() - mp[:, 0].min()) / L, (mp[:, 1].max() - mp[:, 1].min()) / W] if len(mp) >= 2 else [0.0, 0.0]
                opx = np.sort(op[:, 0])[::-1] if len(op) else np.zeros(0)
                off_line = opx[1] / L if len(opx) >= 2 else 1.0
                ol = off_last[t, j] * sgn
                slot = (mc + ol) if np.isfinite(ol).all() and len(mates) else pr
                v3 = vel3[t, j] * sgn
                f += [bv[0] / 10, bv[1] / 10, ext[0], ext[1], off_line, (slot[0] - pr[0]) / 20, (slot[1] - pr[1]) / 20,
                      float(np.isfinite(ol).all()), v3[0] / 8, v3[1] / 8]
                if version >= 3:
                    dead = float(alive is not None and not alive[t])
                    f += [dead, np.log1p(dead_for[t]) / 4, dead * float(bl[0] > 75), dead * float(bl[0] < 30),
                          dead * float(min(bl[1], W - bl[1]) < 3), min(bl[1], W - bl[1]) / (W / 2)]
                if mode == "replay":
                    on_ = off_next[t, j] * sgn
                    slot_n = (mc + on_) if np.isfinite(on_).all() and len(mates) else pr
                    f += [(slot_n[0] - pr[0]) / 20, (slot_n[1] - pr[1]) / 20, float(np.isfinite(on_).all())]
            rows.append(np.concatenate([np.array(f, dtype=np.float32), role_oh[j], np.array(nei, dtype=np.float32)]))
            idx.append((t, j))
    return (np.stack(rows).astype(np.float32) if rows else np.zeros((0, 1), np.float32)), np.array(idx, dtype=np.int32)


# ----------------------------------------------------------------------------- dataset
def _segment_job(args):
    mid, si, mk, seed, stride, version = args
    info, segs = parse_dfl(mid)
    seg = segs[si]
    vis = make_mask(seg, mk, seed)
    on = ~np.isnan(seg["pos"][..., 0])
    out = {}
    for mode in ("live", "replay"):
        prior = formation(info, seg, vis, mode).pos
        X, idx = features(np.where(vis[..., None], seg["pos"], np.nan), vis, on, seg["team"], seg["role"], seg["ball"],
                          seg["poss"], seg["alive"], prior, mode, t_index=range(0, len(vis), stride), version=version)
        if not len(idx):
            continue
        t, j = idx[:, 0], idx[:, 1]
        is_b = (seg["team"][j] == "B")[:, None]
        truth = np.where(is_b, np.stack([L - seg["pos"][t, j, 0], W - seg["pos"][t, j, 1]], 1), seg["pos"][t, j])
        pr = np.where(is_b, np.stack([L - prior[t, j, 0], W - prior[t, j, 1]], 1), prior[t, j])
        out[mode] = {"X": X, "Y": ((truth - pr) / 20).astype(np.float32), "role": seg["role"][j],
                     "mask": np.array([mk] * len(j)), "t": t, "j": j}
    return (mid, si, mk), out


def build_per_match(matches, masks, stride, seed=0, workers=6, kind="train"):
    """Like build(), cached per match so leave-one-match-out folds reuse features."""
    parts = []
    for mid in matches:
        v = "" if FEAT_VERSION == 1 else f"_v{FEAT_VERSION}"
        parts.append(build([mid], masks, stride, seed, workers, tag=f"{kind}{v}_{mid}"))
    return {mode: {k: np.concatenate([p[mode][k] for p in parts]) for k in parts[0][mode]} for mode in ("live", "replay")}


def build(matches, masks, stride, seed=0, workers=6, tag="train"):
    cache = CACHE / f"ml_{tag}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return d["data"].item()
    jobs = []
    for mid in matches:
        _, segs = parse_dfl(mid)
        for si in range(len(segs)):
            for mi, mk in enumerate(masks):
                jobs.append((mid, si, mk, seed + 101 * si + mi, stride, FEAT_VERSION))
    data = {"live": [], "replay": []}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for key, out in ex.map(_segment_job, jobs):
            for mode, d in out.items():
                data[mode].append(d)
    merged = {mode: {k: np.concatenate([d[k] for d in v]) for k in v[0]} for mode, v in data.items()}
    np.savez_compressed(cache, data=np.array(merged, dtype=object))
    return merged


# ----------------------------------------------------------------------------- model
def make_model(n_in):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(nn.Linear(n_in, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
            self.mu = nn.Linear(128, 2)
            self.logsig = nn.Linear(128, 1)

        def forward(self, x):
            h = self.body(x)
            return self.mu(h), self.logsig(h).squeeze(-1)
    torch.manual_seed(0)
    return Net()


def train_model(tr, va, epochs=12, lr=1e-3, bs=2048):
    import torch
    from torch import nn
    Xt, Yt = torch.from_numpy(tr["X"]), torch.from_numpy(tr["Y"])
    Xv, Yv = torch.from_numpy(va["X"]), torch.from_numpy(va["Y"])
    model = make_model(Xt.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state, hist = 1e9, None, []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for k in range(0, len(Xt), bs):
            b = perm[k:k + bs]
            mu, ls = model(Xt[b])
            d2 = ((mu.detach() - Yt[b]) ** 2).sum(1) * 400  # metres^2
            loss = nn.functional.huber_loss(mu, Yt[b], delta=0.25) + 0.01 * (2 * ls + d2 / (2 * torch.exp(2 * ls) * 400)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            mu, _ = model(Xv)
            mae = float(torch.linalg.norm((mu - Yv) * 20, dim=1).mean())
        hist.append(round(mae, 3))
        print(f"  epoch {ep + 1}: val MAE {mae:.2f} m", flush=True)
        if mae < best:
            best, best_state = mae, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, hist


def predict(model, X):
    import torch
    with torch.no_grad():
        mu, ls = model(torch.from_numpy(X))
    return mu.numpy() * 20, np.exp(ls.numpy())  # correction (m), sigma (m-scale units)


def calibrate(sig, err):
    """Empirical: sigma deciles -> r80 and P(err <= 5 m) (fitted on validation only)."""
    edges = np.quantile(sig, np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    tab = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (sig >= a) & (sig < b)
        tab.append({"lo": float(a), "hi": float(b), "r80": float(np.percentile(err[sel], 80)), "p5": float((err[sel] <= 5).mean())})
    return tab


def apply_calibration(tab, sig):
    r = np.zeros_like(sig)
    p = np.zeros_like(sig)
    for row in tab:
        sel = (sig >= row["lo"]) & (sig < row["hi"])
        r[sel], p[sel] = row["r80"], row["p5"]
    return r, p


# ----------------------------------------------------------------------------- experiment
def evaluate_test(models, cals, test_mid, seed=777):
    """All methods on the held-out match, per mask; ML corrections applied on the formation prior."""
    info, segs = parse_dfl(test_mid)
    res, detail = {}, {}
    for mk in EVAL_MASKS:
        errs = {}
        cover = {"live": [], "replay": []}
        for si, seg in enumerate(segs):
            vis = make_mask(seg, mk, seed + si)
            on = ~np.isnan(seg["pos"][..., 0])
            preds = {"last_position": last_position(seg, vis), "constant_velocity": constant_velocity(seg, vis)}
            for mode in ("live", "replay"):
                prior = formation(info, seg, vis, mode).pos
                preds[f"formation_{mode}"] = prior
                X, idx = features(np.where(vis[..., None], seg["pos"], np.nan), vis, on, seg["team"], seg["role"],
                                  seg["ball"], seg["poss"], seg["alive"], prior, mode)
                p = prior.copy()
                if len(idx):
                    corr, sig = predict(models[mode], X)
                    t, j = idx[:, 0], idx[:, 1]
                    is_b = (seg["team"][j] == "B")[:, None]
                    corr = np.where(is_b, -corr, corr)  # own frame -> world
                    p[t, j] = np.clip(prior[t, j] + corr, [0, 0], [L, W])
                    r80, _ = apply_calibration(cals[mode], sig)
                    e_ml = np.linalg.norm(p[t, j] - seg["pos"][t, j], axis=1)
                    cover[mode].append(e_ml <= r80)
                preds[f"ml_{mode}"] = p
            for name, pr in preds.items():
                e, hid = errors(seg, vis, pr)
                errs.setdefault(name, []).append(e[hid])
                if mk == "sc_camera" and name in ("formation_replay", "ml_replay", "ml_live"):
                    detail.setdefault(name, []).append(breakdown(seg, vis, e, hid))
        res[mk] = {n: summarize(np.concatenate(v)) for n, v in errs.items()}
        for mode in ("live", "replay"):
            if cover[mode]:
                res[mk][f"ml_{mode}"]["radius_coverage_80"] = round(float(np.concatenate(cover[mode]).mean()), 3)
        print(f"  {mk}: " + " | ".join(f"{n} {v['mae_m']}" for n, v in res[mk].items()), flush=True)
    return res, detail


def lomo_split(i):
    """Leave-one-match-out over all DFL matches: test = match i, validation = the next match, train = the rest."""
    from backend.reconstruction.datasets import dfl_matches
    ms = dfl_matches()
    test, val = ms[i % len(ms)], ms[(i + 1) % len(ms)]
    return [m for m in ms if m not in (test, val)], val, test


VAL_FROM_T = 2200   # s into each half: the last part of every training half is held back for early stopping / calibration
VAL_GAP_S = 60      # s dropped before it so adjacent seconds never sit on both sides


def _rows(d, sel):
    return {k: v[sel] for k, v in d.items()}


def refit_full(tr_all, va, hist, cal_first, model_first):
    """Retrain on every training row for the epoch count the time-blocked validation picked; carry the out-of-sample
    calibration over by the rank (decile) of the predicted sigma."""
    import torch
    best_ep = int(np.argmin(hist)) + 1
    model = train_model(tr_all, va, best_ep)[0]  # va only printed; selection already done
    _, sig2 = predict(model, va["X"])
    edges = np.quantile(sig2, np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    cal = [{"lo": float(a), "hi": float(b), "r80": c["r80"], "p5": c["p5"]} for a, b, c in zip(edges[:-1], edges[1:], cal_first)]
    return model, cal, best_ep


def run_k7(i, epochs=12, tag="k7", refit=False):
    """7-fold: train on the 6 other DFL matches (all their data except the time-blocked tail of each half, used for
    early stopping and uncertainty calibration), test once on match i."""
    from backend.reconstruction.datasets import dfl_matches
    t0 = time.time()
    ms = dfl_matches()
    test = ms[i]
    train = [m for m in ms if m != test]
    name = f"{tag}_{i}" + ("" if FEAT_VERSION == 1 else f"_v{FEAT_VERSION}")
    print(f"fold {name}: train {train}  test {test}", flush=True)
    data = build_per_match(train, TRAIN_MASKS, stride=2, kind="train")
    import torch
    models, cals, hists = {}, {}, {}
    for mode in ("live", "replay"):
        d = data[mode]
        tr = _rows(d, d["t"] < VAL_FROM_T - VAL_GAP_S)
        va = _rows(d, d["t"] >= VAL_FROM_T)
        print(f"{mode}: train rows {len(tr['X'])}, time-blocked validation rows {len(va['X'])}", flush=True)
        models[mode], hists[mode] = train_model(tr, va, epochs)
        corr, sig = predict(models[mode], va["X"])
        err = np.linalg.norm(corr - va["Y"] * 20, axis=1)
        cals[mode] = calibrate(sig, err)
        if refit:
            models[mode], cals[mode], ep = refit_full(d, va, hists[mode], cals[mode], models[mode])
            print(f"{mode}: refit on all rows of the 6 matches for {ep} epochs", flush=True)
        WEIGHTS.mkdir(exist_ok=True)
        torch.save({"state": models[mode].state_dict(), "n_in": va["X"].shape[1], "calibration": cals[mode],
                    "feat_version": FEAT_VERSION, "mode": mode, "fold": name, "train": train, "test": test},
                   WEIGHTS / f"hidden_{mode}_{name}.pt")
    res, detail = evaluate_test(models, cals, test)
    out = {"fold": name, "train": train, "val": f"time-blocked: t >= {VAL_FROM_T}s of each training half", "refit_on_all": refit, "test_match": test,
           "val_curve_mae_m": hists, "results": res, "test_breakdown_sc_camera": detail, "seconds": round(time.time() - t0)}
    json.dump(out, open(CACHE / f"ml_results_{name}.json", "w"), indent=1, default=str)
    return out


def run(fold=0, epochs=12, lomo=False):
    t0 = time.time()
    train, val, test = lomo_split(fold) if lomo else split(fold)
    name = (f"lomo{fold}" if lomo else f"f{fold}") + ("" if FEAT_VERSION == 1 else f"_v{FEAT_VERSION}")
    print(f"fold {name}: train {train}  val {val}  test {test}", flush=True)
    if lomo:
        tr = build_per_match(train, TRAIN_MASKS, stride=2, kind="train")
        va = build_per_match([val], ["sc_camera", "k=14", "k=10", "k=6"], stride=3, seed=55, kind="val")
    else:
        tr = build(train, TRAIN_MASKS, stride=2, tag=f"train_f{fold}")
        va = build([val], ["sc_camera", "k=14", "k=10", "k=6"], stride=3, seed=55, tag=f"val_f{fold}")
    print(f"features: train live {len(tr['live']['X'])}, replay {len(tr['replay']['X'])}; val {len(va['live']['X'])} ({time.time() - t0:.0f}s)", flush=True)
    import torch
    models, cals, hists = {}, {}, {}
    for mode in ("live", "replay"):
        print(f"training {mode} model", flush=True)
        models[mode], hists[mode] = train_model(tr[mode], va[mode], epochs)
        corr, sig = predict(models[mode], va[mode]["X"])
        err = np.linalg.norm(corr - va[mode]["Y"] * 20, axis=1)
        cals[mode] = calibrate(sig, err)
        WEIGHTS.mkdir(exist_ok=True)
        torch.save({"state": models[mode].state_dict(), "n_in": va[mode]["X"].shape[1], "calibration": cals[mode],
                    "feat_version": FEAT_VERSION,
                    "mode": mode, "fold": fold, "train": train, "val": val, "test": test}, WEIGHTS / f"hidden_{mode}_{name}.pt")
    print("evaluating on held-out match", flush=True)
    res, detail = evaluate_test(models, cals, test)
    out = {"fold": fold, "train": train, "val": val, "test_match": test, "val_curve_mae_m": hists, "results": res,
           "test_breakdown_sc_camera": detail, "seconds": round(time.time() - t0)}
    json.dump(out, open(CACHE / f"ml_results_{name}.json", "w"), indent=1, default=str)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lomo", action="store_true", help="leave-one-match-out fold (0..6) over all DFL matches")
    ap.add_argument("--all", action="store_true", help="with --lomo: run every fold")
    ap.add_argument("--feat", type=int, default=1, help="feature version (2 = temporal / team-relative)")
    ap.add_argument("--k7", action="store_true", help="7-fold: train on 6 matches, test on the 7th (--all for every fold)")
    ap.add_argument("--tag", default="k7")
    ap.add_argument("--refit", action="store_true", help="7-fold: refit on all 6 matches after epoch selection")
    a = ap.parse_args()
    FEAT_VERSION = a.feat
    if a.k7:
        for i in (range(7) if a.all else [a.fold]):
            run_k7(i, a.epochs, a.tag, a.refit)
        raise SystemExit
    if a.lomo and a.all:
        from backend.reconstruction.datasets import dfl_matches
        for i in range(len(dfl_matches())):
            run(i, a.epochs, lomo=True)
    else:
        run(a.fold, a.epochs, lomo=a.lomo)


# ----------------------------------------------------------------------------- synthetic benchmark (cross-domain)
SIM_COARSE = {"GK": "GK", "RB": "FB", "LB": "FB", "RCB": "CB", "LCB": "CB", "CM": "CM", "RCM": "CM", "LCM": "CM",
              "RDM": "DM", "LDM": "DM", "CAM": "AM", "RAM": "W", "LAM": "W", "RW": "W", "LW": "W", "RM": "WM", "LM": "WM",
              "ST": "ST", "RS": "ST", "LS": "ST"}


def _sim_job(args):
    seed, masks, fold = args
    from backend.models.match import MatchData
    from backend.reconstruction.estimator import reconstruct, roster_timeline
    from backend.reconstruction.inference import reconstruct_ml
    from backend.reconstruction.observation import observe
    from simulator.match_generator import generate_match
    m = MatchData(generate_match(seed))
    on, role = roster_timeline(m)
    teams = np.array([p[0] for p in m.order])
    rc = np.array([SIM_COARSE.get(role.get(p) or "CM", "CM") for p in m.order])
    out = {}
    for mk in masks:
        vis = observe(m, mk, seed).vis & on
        hid = on & ~vis & ~np.isnan(m.xy[..., 0])
        for mode in ("live", "replay"):
            for name, rec in (("formation", reconstruct(m, vis, mode, {})),
                              ("ml", reconstruct_ml(m, vis, mode, on, role, None, teams, rc, fold=fold))):
                e = np.linalg.norm(rec.pos - m.xy, axis=2)[hid]
                out.setdefault(mk, {})[f"{name}_{mode}"] = e
    return out


def eval_synthetic(n=20, masks=("broadcast", "k=14", "k=10", "k=6"), fold=0, workers=6):
    seeds = [s for s in range(1, n + 2) if s != 11][:n]
    acc = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for out in ex.map(_sim_job, [(s, masks, fold) for s in seeds]):
            for mk, d in out.items():
                for k, e in d.items():
                    acc.setdefault(mk, {}).setdefault(k, []).append(e)
    res = {mk: {k: summarize(np.concatenate(v)) for k, v in d.items()} for mk, d in acc.items()}
    json.dump({"matches": len(seeds), "fold": fold, "results": res}, open(CACHE / f"ml_synthetic_f{fold}.json", "w"), indent=1)
    return res


def run_final(epochs=12, tag="final", feat=None):
    """Production model: trained on all 7 DFL matches with the 7-fold recipe (time-blocked tail of every half for
    epoch choice + calibration, then refit on everything). Its accuracy is the 7-fold estimate; it has no test match."""
    import torch
    from backend.reconstruction.datasets import dfl_matches
    global FEAT_VERSION
    if feat is not None:
        FEAT_VERSION = feat
    ms = dfl_matches()
    data = build_per_match(ms, TRAIN_MASKS, stride=2, kind="train")
    name = f"{tag}" + ("" if FEAT_VERSION == 1 else f"_v{FEAT_VERSION}")
    for mode in ("live", "replay"):
        d = data[mode]
        tr = _rows(d, d["t"] < VAL_FROM_T - VAL_GAP_S)
        va = _rows(d, d["t"] >= VAL_FROM_T)
        model, hist = train_model(tr, va, epochs)
        corr, sig = predict(model, va["X"])
        cal = calibrate(sig, np.linalg.norm(corr - va["Y"] * 20, axis=1))
        model, cal, ep = refit_full(d, va, hist, cal, model)
        torch.save({"state": model.state_dict(), "n_in": va["X"].shape[1], "calibration": cal, "feat_version": FEAT_VERSION,
                    "mode": mode, "fold": name, "train": ms, "test": None, "epochs": ep}, WEIGHTS / f"hidden_{mode}_{name}.pt")
        print(f"saved hidden_{mode}_{name}.pt ({ep} epochs)", flush=True)
