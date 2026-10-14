"""Train hidden-player models under the 7-fold match-level protocol (DFL truth, real SkillCorner camera masks).

Outer fold i: test match = dfl_matches()[i], never read here. Training = the other 6 matches. Model selection
(epoch, and later the choice between architectures) uses only the time-blocked tail of every training half
(t >= 2200 s, 60 s gap before it), never the test match. Uncertainty is calibrated on that same tail.

    .venv/bin/python -m backend.reconstruction.train_hidden --arch mlp --fold all --tag r13_mlp
    .venv/bin/python -m backend.reconstruction.train_hidden --arch gru_ctx --fold all --tag r13_gru
    .venv/bin/python -m backend.reconstruction.train_hidden --arch mlp --fold final --tag r13_mlp   # all 7 matches

Checkpoints: cv/weights/recon/<tag>/hidden_<mode>_fold<i>.pt (+ train_log.json). Features: ml.features v3 rows +
hidden_models.context_tensor, built per (match, half, mask instance) and cached in data/eval/real/recon_cache.
"""
import argparse
import hashlib
import inspect
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from backend.reconstruction import hidden_models as hm
from backend.reconstruction.datasets import CACHE, L, W, dfl_matches, parse_dfl
from backend.reconstruction.ml import calibrate, features
from backend.reconstruction.real_benchmark import formation, make_mask

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "cv" / "weights" / "recon"
DCACHE = CACHE / "recon_cache"
TRAIN_MASKS = ["sc_camera", "sc_camera", "k=16", "k=12", "k=8", "k=6", "rand=10"]  # same recipe as ml.py k7r
FEAT = 3
STRIDE = 2
VAL_FROM_T, VAL_GAP_S = 2200, 60
TRAIN_SEED = 4242  # mask seeds for training instances; evaluation uses benchmark.EVAL_SEED (disjoint)


def _source_digest():
    src = inspect.getsource(features) + inspect.getsource(hm.context_tensor)
    for f in ("ml.py", "real_benchmark.py", "estimator.py", "datasets.py", "hidden_models.py"):
        src += (Path(__file__).with_name(f)).read_text()
    return hashlib.sha256(src.encode()).hexdigest()[:10]


def _instance_job(args):
    mid, si, mk, seed = args
    info, segs = parse_dfl(mid)
    seg = segs[si]
    vis = make_mask(seg, mk, seed)
    on = ~np.isnan(seg["pos"][..., 0])
    pos_obs = np.where(vis[..., None], seg["pos"], np.nan)
    out = {}
    for mode in ("live", "replay"):
        prior = formation(info, seg, vis, mode).pos
        X, idx = features(pos_obs, vis, on, seg["team"], seg["role"], seg["ball"], seg["poss"], seg["alive"], prior,
                          mode, t_index=range(0, len(vis), STRIDE), version=FEAT)
        C = hm.context_tensor(pos_obs, vis, on, seg["team"], seg["ball"], seg["poss"], seg["alive"], prior)
        t, j = idx[:, 0], idx[:, 1]
        is_b = (seg["team"][j] == "B")[:, None]
        truth = np.where(is_b, np.stack([L - seg["pos"][t, j, 0], W - seg["pos"][t, j, 1]], 1), seg["pos"][t, j])
        pr = np.where(is_b, np.stack([L - prior[t, j, 0], W - prior[t, j, 1]], 1), prior[t, j])
        out[mode] = {"X": X, "Y": ((truth - pr) / 20).astype(np.float32), "t": t.astype(np.int32),
                     "j": j.astype(np.int16), "C": C.astype(np.float16)}
    return (mid, si, mk, seed), out


DEG_WINDOW = 30  # s: independent broadcast-like clips for context-degradation augmentation (E7)


def _degraded_job(args):
    """GSR-like context on DFL: independent DEG_WINDOW-s clips, no possession / ball status, GK-vs-outfield roles only,
    no DFL anchors. One instance per clip; rows keep their absolute time in 'tabs' for the time-blocked split."""
    from types import SimpleNamespace
    from backend.reconstruction.estimator import reconstruct
    from backend.reconstruction.real_benchmark import mask_sc
    mid, si, seed = args
    _, segs = parse_dfl(mid)
    seg = segs[si]
    on_all = ~np.isnan(seg["pos"][..., 0])
    vis_all = mask_sc(seg, seed) & on_all
    team = seg["team"]
    role = np.where(seg["role"] == "GK", "GK", "CM")
    order = [f"{tm}:{j}" for j, tm in enumerate(team)]
    out = {"live": [], "replay": []}
    for t0 in range(0, len(vis_all) - DEG_WINDOW + 1, DEG_WINDOW):
        sl = slice(t0, t0 + DEG_WINDOW)
        pos, vis, on, ball = seg["pos"][sl], vis_all[sl], on_all[sl], seg["ball"][sl]
        T = DEG_WINDOW
        poss, alive = np.full(T, -1), np.ones(T, bool)
        pos_obs = np.where(vis[..., None], pos, np.nan)
        mm = SimpleNamespace(xy=pos_obs, ball=ball, poss=poss, order=order, T=T, idx={o: j for j, o in enumerate(order)},
                             players=[], events=[])
        for mode in ("live", "replay"):
            prior = reconstruct(mm, vis, mode, {}, on=on, role={o: r for o, r in zip(order, role)}, anchors=None).pos
            X, idx = features(pos_obs, vis, on, team, role, ball, poss, alive, prior, mode,
                              t_index=range(0, T, STRIDE), version=FEAT)
            if not len(idx):
                continue
            t, j = idx[:, 0], idx[:, 1]
            is_b = (team[j] == "B")[:, None]
            truth = np.where(is_b, np.stack([L - pos[t, j, 0], W - pos[t, j, 1]], 1), pos[t, j])
            pr = np.where(is_b, np.stack([L - prior[t, j, 0], W - prior[t, j, 1]], 1), prior[t, j])
            C = hm.context_tensor(pos_obs, vis, on, team, ball, poss, alive, prior)
            out[mode].append({"X": X, "Y": ((truth - pr) / 20).astype(np.float32), "t": t.astype(np.int32),
                              "tabs": (t + t0).astype(np.int32), "j": j.astype(np.int16), "C": C.astype(np.float16)})
    return (mid, si), out


def degraded_data(mid, workers=8):
    DCACHE.mkdir(parents=True, exist_ok=True)
    f = DCACHE / f"{mid}_degraded_w{DEG_WINDOW}_s{STRIDE}_{_source_digest()}.npz"
    if f.exists():
        return np.load(f, allow_pickle=True)["d"].item()
    _, segs = parse_dfl(mid)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_degraded_job, [(mid, si, TRAIN_SEED + 7 + 101 * si) for si in range(len(segs))]))
    d = {"modes": {mode: {"rows": [r | {"C": None} for _, o in res for r in o[mode]],
                          "C": [r["C"] for _, o in res for r in o[mode]],
                          "mask": ["degraded"] * sum(len(o[mode]) for _, o in res)} for mode in ("live", "replay")}}
    np.savez(f, d=np.array(d, dtype=object))
    return d


def match_data(mid, workers=8):
    """All training instances of one match (both halves x TRAIN_MASKS), cached on disk."""
    DCACHE.mkdir(parents=True, exist_ok=True)
    f = DCACHE / f"{mid}_train_s{STRIDE}_{_source_digest()}.npz"
    if f.exists():
        return np.load(f, allow_pickle=True)["d"].item()
    _, segs = parse_dfl(mid)
    jobs = [(mid, si, mk, TRAIN_SEED + 101 * si + mi) for si in range(len(segs)) for mi, mk in enumerate(TRAIN_MASKS)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_instance_job, jobs))
    d = {"instances": [k for k, _ in res], "modes": {}}
    for mode in ("live", "replay"):
        d["modes"][mode] = {"rows": [o[mode] | {"C": None} for _, o in res], "C": [o[mode]["C"] for _, o in res],
                            "mask": [k[2] for k, _ in res]}
    np.savez(f, d=np.array(d, dtype=object))
    return d


def assemble(mids, mode, augment=False):
    """Rows + context store over several matches. inst indexes the concatenated context tensors. 't' is the index
    into the instance's context; 'tabs' the absolute second in the half (used for the time-blocked split)."""
    X, Y, T, TA, J, I, MK, C = [], [], [], [], [], [], [], []
    for mid in mids:
        sources = [match_data(mid)["modes"][mode]] + ([degraded_data(mid)["modes"][mode]] if augment else [])
        for d in sources:
            for r, c, mk in zip(d["rows"], d["C"], d["mask"]):
                k = len(C)
                C.append(c)
                X.append(r["X"]), Y.append(r["Y"]), T.append(r["t"]), J.append(r["j"]), TA.append(r.get("tabs", r["t"]))
                I.append(np.full(len(r["t"]), k, np.int32)), MK.append(np.full(len(r["t"]), mk))
    return {"X": np.concatenate(X), "Y": np.concatenate(Y), "t": np.concatenate(T).astype(np.int64),
            "tabs": np.concatenate(TA).astype(np.int64),
            "j": np.concatenate(J).astype(np.int64), "inst": np.concatenate(I).astype(np.int64),
            "mask": np.concatenate(MK), "store": hm.CtxStore(C, mode)}


def _sel(d, s):
    return {k: (v[s] if isinstance(v, np.ndarray) else v) for k, v in d.items()}


def train(arch, tr, va, mode, epochs, seed=0, lr=1e-3, bs=2048, device="cpu", log=print):
    import torch
    from torch import nn
    from backend.reconstruction.ml import make_model
    n_in = tr["X"].shape[1]
    torch.manual_seed(seed)
    net = make_model(n_in) if arch == "mlp" else hm.make_gru(n_in, mode)
    if seed:  # make_* seed 0 internally; re-initialise for other seeds
        torch.manual_seed(seed)
        for m in net.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
    net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    rng = np.random.default_rng(seed)

    def batch(d, b):
        x = torch.from_numpy(d["X"][b]).to(device)
        y = torch.from_numpy(d["Y"][b]).to(device)
        s = torch.from_numpy(d["store"].windows(d["inst"][b], d["t"][b], d["j"][b])).to(device) if arch == "gru_ctx" else None
        return x, y, s

    def fwd(x, s):
        return net(x, s) if arch == "gru_ctx" else net(x)

    def evaluate(d):
        net.eval()
        mus, sigs = [], []
        with torch.no_grad():
            for k in range(0, len(d["X"]), 8192):
                b = np.arange(k, min(k + 8192, len(d["X"])))
                x, _, s = batch(d, b)
                mu, ls = fwd(x, s)
                mus.append(mu.cpu().numpy() * 20)
                sigs.append(np.exp(ls.cpu().numpy()))
        return np.concatenate(mus), np.concatenate(sigs)

    sc = va["mask"] == "sc_camera"
    deg = va["mask"] == "degraded"
    best, best_state, hist = 1e9, None, []
    for ep in range(epochs):
        t0 = time.time()
        net.train()
        perm = rng.permutation(len(tr["X"]))
        for k in range(0, len(perm), bs):
            b = np.sort(perm[k:k + bs])
            x, y, s = batch(tr, b)
            mu, ls = fwd(x, s)
            d2 = ((mu.detach() - y) ** 2).sum(1) * 400
            loss = nn.functional.huber_loss(mu, y, delta=0.25) + 0.01 * (2 * ls + d2 / (2 * torch.exp(2 * ls) * 400)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        mu, _ = evaluate(va)
        err = np.linalg.norm(mu - va["Y"] * 20, axis=1)
        h = {"epoch": ep + 1, "val_mae_all": round(float(err.mean()), 4), "val_mae_sc_camera": round(float(err[sc].mean()), 4),
             "seconds": round(time.time() - t0, 1)}
        crit = h["val_mae_sc_camera"]
        if deg.any():  # E7 selection rule: mean of full-context and degraded-context validation error
            h["val_mae_degraded"] = round(float(err[deg].mean()), 4)
            crit = (h["val_mae_sc_camera"] + h["val_mae_degraded"]) / 2
        h["criterion"] = round(crit, 4)
        hist.append(h)
        log(f"    {mode} ep {ep + 1}: val {h['val_mae_all']:.3f} m (sc_camera {h['val_mae_sc_camera']:.3f}"
            + (f", degraded {h['val_mae_degraded']:.3f}" if deg.any() else "") + f") {h['seconds']}s")
        if crit < best:
            best, best_state = crit, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.to("cpu")
    mu, sig = None, None
    net.eval()
    device_saved = device
    device = "cpu"
    mu, sig = evaluate(va)
    device = device_saved
    err = np.linalg.norm(mu - va["Y"] * 20, axis=1)
    val = {"val_mae_sc_camera": float(err[sc].mean()), "val_mae_all": float(err.mean()), "criterion": best}
    if deg.any():
        val["val_mae_degraded"] = float(err[deg].mean())
    return net, hist, calibrate(sig, err), val


def fold_split(fold, ms=None):
    """(training matches, test match). 'final' trains on every match and has no test match."""
    ms = list(ms if ms is not None else dfl_matches())
    if fold == "final":
        return ms, None
    test = ms[int(fold)]
    return [m for m in ms if m != test], test


def run_fold(arch, fold, tag, epochs, seeds=(0,), device="cpu", augment=False):
    import torch
    train_ms, test = fold_split(fold)
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / f"train_fold{fold}.log", "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
    log(f"[{time.strftime('%H:%M:%S')}] {arch} fold {fold}: train {train_ms} test {test} seeds {seeds}")
    summary = {"arch": arch, "fold": fold, "train": train_ms, "test": test, "modes": {}}
    for mode in ("live", "replay"):
        d = assemble(train_ms, mode, augment)
        tr = _sel(d, d["tabs"] < VAL_FROM_T - VAL_GAP_S)
        va = _sel(d, d["tabs"] >= VAL_FROM_T)
        log(f"  {mode}: train rows {len(tr['X'])}, time-blocked validation rows {len(va['X'])}")
        names = []
        for seed in seeds:
            net, hist, cal, val = train(arch, tr, va, mode, epochs, seed=seed, device=device, log=log)
            name = f"hidden_{mode}_fold{fold}" + (f"_s{seed}" if len(seeds) > 1 else "") + ".pt"
            torch.save({"arch": arch, "state": net.state_dict(), "n_in": tr["X"].shape[1], "calibration": cal,
                        "feat_version": FEAT, "mode": mode, "fold": fold, "train": train_ms, "test": test, "seed": seed,
                        "epochs_run": epochs, "best_epoch": int(np.argmin([h["val_mae_sc_camera"] for h in hist])) + 1,
                        "history": hist, "val": val, "window": hm.window_offsets(mode).tolist(),
                        "protocol": "7-fold match-level; selection + calibration on time-blocked tail of training halves",
                        "augment": "degraded broadcast context (30 s clips, no possession/status/roles)" if augment else None,
                        "source_digest": _source_digest()}, out / name)
            names.append(name)
            summary["modes"].setdefault(mode, []).append({"seed": seed, "file": name, "val": val, "history": hist})
        if len(names) > 1:  # ensemble checkpoint: recalibrate the combined sigma on the same validation tail
            members = [hm.load_hidden(out / n) for n in names]
            ens = hm.HiddenModel({"arch": "ensemble", "mode": mode, "calibration": None, "feat_version": FEAT}, members)
            seq = va["store"].windows(va["inst"], va["t"], va["j"]) if ens.needs_context else None
            mu, sig = ens.raw(va["X"], seq)
            err = np.linalg.norm(mu - va["Y"] * 20, axis=1)
            torch.save({"arch": "ensemble", "members": names, "mode": mode, "feat_version": FEAT, "fold": fold,
                        "train": train_ms, "test": test, "calibration": calibrate(sig, err),
                        "val": {"val_mae_all": float(err.mean()), "val_mae_sc_camera": float(err[va["mask"] == "sc_camera"].mean()),
                                **({"val_mae_degraded": float(err[va["mask"] == "degraded"].mean())} if (va["mask"] == "degraded").any() else {})}},
                       out / f"hidden_{mode}_fold{fold}.pt")
            summary["modes"][mode].append({"ensemble": True, "val_mae_sc_camera": float(err[va["mask"] == "sc_camera"].mean())})
            log(f"  {mode} ensemble val sc_camera {err[va['mask'] == 'sc_camera'].mean():.3f}")
    (out / f"summary_fold{fold}.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["mlp", "gru_ctx"], default="mlp")
    ap.add_argument("--fold", default="all", help="0..6, 'all', or 'final' (all 7 matches, no test)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--augment", action="store_true", help="E7: add degraded-context 30 s clips to training and validation")
    a = ap.parse_args()
    if a.build_only:
        for mid in dfl_matches():
            t0 = time.time()
            match_data(mid)
            if a.augment:
                degraded_data(mid)
            print(mid, f"{time.time() - t0:.0f}s", flush=True)
        raise SystemExit
    seeds = tuple(int(s) for s in a.seeds.split(","))
    folds = range(7) if a.fold == "all" else [a.fold if a.fold == "final" else int(a.fold)]
    for f in folds:
        run_fold(a.arch, f, a.tag, a.epochs, seeds, a.device, a.augment)


def validation_table(tags, folds=range(7)):
    """Model selection evidence: per variant and mode, the validation-tail MAE (sc_camera rows of the time-blocked
    tail of the TRAINING matches) of every fold, never test error. Single models: best epoch; ensembles: combined."""
    import torch
    out = {}
    for tag in tags:
        for mode in ("live", "replay"):
            vals = []
            for f in folds:
                p = OUT / tag / f"hidden_{mode}_fold{f}.pt"
                if not p.exists():
                    vals = None
                    break
                ck = torch.load(p, weights_only=False)
                vals.append(float(ck["val"]["val_mae_sc_camera"]))
            if vals:
                out[f"{tag}:{mode}"] = {"per_fold": [round(v, 3) for v in vals], "mean": round(float(np.mean(vals)), 3)}
    return out
