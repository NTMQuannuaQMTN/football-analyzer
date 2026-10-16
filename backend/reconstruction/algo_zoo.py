"""Compare machine-learning algorithm families on the hidden-player position task, under the 7-fold protocol.

Task (identical for every algorithm): from the v3 row features of a hidden player-second (66 live / 75 replay), predict
the 2-D correction to the team-shape prior (own-team frame, metres). Training rows: the 6 training matches of a fold,
with context-degradation augmentation (as the production model). Screening score: mean position error on the
time-blocked validation tail of those TRAINING matches, separately for full-context (real camera footprint) and
degraded broadcast-like rows; criterion = mean of the two (the declared selection rule). Test matches are not used.

Families
  zero            no correction (the team-shape prior itself)
  ridge           linear regression, L2 (closed form)
  linear_svr      linear support-vector regression (epsilon-insensitive), one model per axis
  knn             k-nearest-neighbour regression (k=25, distance weighted, standardised features)
  decision_tree   single CART regression tree
  random_forest   bagged CART trees
  extra_trees     extremely randomised trees
  hist_gbm        histogram gradient boosting (LightGBM-style), one model per axis
  mlp             multilayer perceptron (the production architecture, single seed)
  mdn             mixture density network (3 Gaussians; predicts the mixture mean)
  zone_classifier classification: softmax over 4 m x 4 m cells of the correction, expected cell centre
  jepa            JEPA-style self-supervised encoder (masked features -> predict EMA target embedding), frozen,
                  then a regression head
Not run (cannot train on ~1.3 M rows here): exact Gaussian processes and kernel SVMs (O(n^2)-O(n^3) memory / time).

    .venv/bin/python -m backend.reconstruction.algo_zoo --folds 0,3,6
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from backend.reconstruction.train_hidden import OUT as WEIGHT_OUT, VAL_FROM_T, VAL_GAP_S, _sel, assemble, fold_split

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "data" / "eval" / "recon_benchmark" / "algo_zoo"
SEED = 16
N_VAL = 150_000          # validation rows scored per fold and mode (same fixed subset for every algorithm)
CAP = {"knn": 150_000, "random_forest": 300_000, "extra_trees": 300_000, "linear_svr": 400_000}  # train-row caps


# ----------------------------------------------------------------------------- torch models
def _torch_fit(model_fn, loss_fn, Xtr, Ytr, epochs=10, bs=2048, lr=1e-3, seed=SEED):
    import torch
    torch.manual_seed(seed)
    net = model_fn()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    X, Y = torch.from_numpy(Xtr), torch.from_numpy(Ytr)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        net.train()
        perm = rng.permutation(len(X))
        for k in range(0, len(perm), bs):
            b = perm[k:k + bs]
            loss = loss_fn(net, X[b], Y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    net.eval()
    return net


def _mlp(n_in, n_out):
    from torch import nn
    return nn.Sequential(nn.Linear(n_in, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
                         nn.Linear(128, n_out))


def fit_mlp(Xtr, Ytr):
    from torch import nn
    net = _torch_fit(lambda: _mlp(Xtr.shape[1], 2), lambda n, x, y: nn.functional.huber_loss(n(x), y, delta=0.25), Xtr, Ytr)
    return lambda X: _torch_predict(net, X)


def _torch_predict(net, X, post=None):
    import torch
    out = []
    with torch.no_grad():
        for k in range(0, len(X), 16384):
            o = net(torch.from_numpy(X[k:k + 16384]))
            out.append((post(o) if post else o).numpy())
    return np.concatenate(out)


def fit_mdn(Xtr, Ytr, K=3):
    import torch
    from torch import nn

    def make():
        return _mlp(Xtr.shape[1], K * 5)  # per component: logit, mu_x, mu_y, log_sx, log_sy

    def split(o):
        o = o.view(-1, K, 5)
        return o[..., 0], o[..., 1:3], o[..., 3:5].clamp(-6, 3)

    def loss(n, x, y):
        logit, mu, ls = split(n(x))
        z = (y[:, None, :] - mu) / torch.exp(ls)
        logp = -0.5 * (z ** 2).sum(-1) - ls.sum(-1) + torch.log_softmax(logit, -1)
        return -torch.logsumexp(logp, -1).mean()
    net = _torch_fit(make, loss, Xtr, Ytr)

    def post(o):
        logit, mu, _ = split(o)
        return (torch.softmax(logit, -1)[..., None] * mu).sum(1)
    return lambda X: _torch_predict(net, X, post)


ZONE_M, ZONE_R = 4.0, 40.0   # 4 m cells over corrections of +-40 m (clipped)


def fit_zone(Xtr, Ytr):
    import torch
    from torch import nn
    nb = int(2 * ZONE_R / ZONE_M)
    cells = np.clip(((Ytr * 20 + ZONE_R) / ZONE_M).astype(int), 0, nb - 1)
    lab = (cells[:, 0] * nb + cells[:, 1]).astype(np.int64)
    centres = torch.from_numpy(np.array([[(i + 0.5) * ZONE_M - ZONE_R, (k + 0.5) * ZONE_M - ZONE_R]
                                         for i in range(nb) for k in range(nb)], np.float32) / 20)
    net = _torch_fit(lambda: _mlp(Xtr.shape[1], nb * nb), lambda n, x, y: nn.functional.cross_entropy(n(x), y),
                     Xtr, lab)
    return lambda X: _torch_predict(net, X, lambda o: torch.softmax(o, -1) @ centres)


def fit_jepa(Xtr, Ytr, emb=128, pre_epochs=6, mask_p=0.3):
    """JEPA-style (feature space): an online encoder sees features with a random 30 % masked, a predictor maps its
    embedding to the embedding an EMA target encoder gives the full features (no reconstruction of inputs). The
    encoder is then frozen and a regression head is trained on the embeddings."""
    import copy

    import torch
    from torch import nn
    torch.manual_seed(SEED)
    n_in = Xtr.shape[1]
    enc = nn.Sequential(nn.Linear(2 * n_in, 256), nn.ReLU(), nn.Linear(256, emb))
    tgt = copy.deepcopy(enc)
    for p in tgt.parameters():
        p.requires_grad_(False)
    pred = nn.Sequential(nn.Linear(emb, 256), nn.ReLU(), nn.Linear(256, emb))
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()), lr=1e-3, weight_decay=1e-4)
    X = torch.from_numpy(Xtr)
    rng = np.random.default_rng(SEED)
    full_mask = torch.zeros_like(X[:1])
    for _ in range(pre_epochs):
        perm = rng.permutation(len(X))
        for k in range(0, len(perm), 2048):
            x = X[perm[k:k + 2048]]
            m = (torch.rand_like(x) < mask_p).float()
            ctx = enc(torch.cat([x * (1 - m), m], 1))
            with torch.no_grad():
                t = tgt(torch.cat([x, full_mask.expand_as(x)], 1))
            loss = nn.functional.smooth_l1_loss(pred(ctx), nn.functional.layer_norm(t, t.shape[-1:]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                for pt, po in zip(tgt.parameters(), enc.parameters()):
                    pt.mul_(0.996).add_(0.004 * po)
    enc.eval()

    def embed(A):
        with torch.no_grad():
            a = torch.from_numpy(A)
            return torch.cat([enc(torch.cat([a[k:k + 16384], full_mask.expand_as(a[k:k + 16384])], 1))
                              for k in range(0, len(a), 16384)]).numpy()
    E = embed(Xtr)
    head = _torch_fit(lambda: _mlp(emb, 2), lambda n, x, y: nn.functional.huber_loss(n(x), y, delta=0.25), E, Ytr)
    return lambda A: _torch_predict(head, embed(A))


# ----------------------------------------------------------------------------- sklearn / numpy models
def _std(Xtr):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    return lambda A: ((A - mu) / sd).astype(np.float32)


def fit_ridge(Xtr, Ytr, alpha=1.0):
    s = _std(Xtr)
    A = np.c_[s(Xtr), np.ones(len(Xtr), np.float32)]
    W = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ Ytr)
    return lambda X: np.c_[s(X), np.ones(len(X), np.float32)] @ W


def fit_sklearn(kind, Xtr, Ytr):
    from sklearn import ensemble, neighbors, svm, tree
    s = _std(Xtr) if kind in ("knn", "linear_svr") else (lambda A: A)
    Z = s(Xtr)
    if kind == "knn":
        m = neighbors.KNeighborsRegressor(n_neighbors=25, weights="distance", n_jobs=-1).fit(Z, Ytr)
    elif kind == "decision_tree":
        m = tree.DecisionTreeRegressor(max_depth=14, min_samples_leaf=50, random_state=SEED).fit(Z, Ytr)
    elif kind == "random_forest":
        m = ensemble.RandomForestRegressor(n_estimators=100, max_depth=18, min_samples_leaf=20, max_features=0.33,
                                           n_jobs=-1, random_state=SEED).fit(Z, Ytr)
    elif kind == "extra_trees":
        m = ensemble.ExtraTreesRegressor(n_estimators=100, max_depth=20, min_samples_leaf=20, max_features=0.5,
                                         n_jobs=-1, random_state=SEED).fit(Z, Ytr)
    elif kind in ("hist_gbm", "linear_svr"):
        ms = []
        for a in range(2):
            if kind == "hist_gbm":
                ms.append(ensemble.HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                                                                 loss="absolute_error", early_stopping=False,
                                                                 random_state=SEED).fit(Z, Ytr[:, a]))
            else:
                ms.append(svm.LinearSVR(epsilon=0.05, C=0.1, max_iter=3000, random_state=SEED).fit(Z, Ytr[:, a]))
        return lambda X: np.stack([mm.predict(s(X)) for mm in ms], 1), ms
    else:
        raise ValueError(kind)
    return lambda X: m.predict(s(X)), m


FITTERS = {"zero": lambda X, Y: (lambda A: np.zeros((len(A), 2), np.float32)),
           "ridge": fit_ridge, "mlp": fit_mlp, "mdn": fit_mdn, "zone_classifier": fit_zone, "jepa": fit_jepa}
SKLEARN = ["linear_svr", "knn", "decision_tree", "random_forest", "extra_trees", "hist_gbm"]
ALL = ["zero", "ridge", "linear_svr", "knn", "decision_tree", "random_forest", "extra_trees", "hist_gbm", "mlp", "mdn",
       "zone_classifier", "jepa"]


def screen(fold, algos, modes=("live", "replay")):
    train_ms, _ = fold_split(fold)
    rng = np.random.default_rng(SEED + fold)
    res = {}
    for mode in modes:
        d = assemble(train_ms, mode, augment=True)
        tr = _sel(d, d["tabs"] < VAL_FROM_T - VAL_GAP_S)
        va = _sel(d, d["tabs"] >= VAL_FROM_T)
        vi = np.sort(rng.choice(len(va["X"]), min(N_VAL, len(va["X"])), replace=False))
        Xv, Yv, mk = va["X"][vi], va["Y"][vi], va["mask"][vi]
        for a in algos:
            cap = CAP.get(a)
            ti = np.sort(rng.choice(len(tr["X"]), cap, replace=False)) if cap and cap < len(tr["X"]) else slice(None)
            Xt, Yt = tr["X"][ti], tr["Y"][ti]
            t0 = time.time()
            fit = FITTERS[a](Xt, Yt) if a in FITTERS else fit_sklearn(a, Xt, Yt)[0]
            t1 = time.time()
            P = fit(Xv)
            t2 = time.time()
            err = np.linalg.norm((P - Yv) * 20, axis=1)
            full, deg = float(err[mk == "sc_camera"].mean()), float(err[mk == "degraded"].mean())
            r = {"val_full_m": round(full, 3), "val_degraded_m": round(deg, 3), "criterion": round((full + deg) / 2, 3),
                 "within_5m": round(float((err <= 5).mean()), 3), "train_rows": int(len(Xt)),
                 "train_s": round(t1 - t0, 1), "predict_ms_per_1k": round(1000 * (t2 - t1) / len(Xv) * 1000, 2)}
            res.setdefault(a, {})[mode] = r
            print(f"fold {fold} {mode:6s} {a:16s} full {full:6.2f}  degraded {deg:6.2f}  crit {r['criterion']:6.2f}  "
                  f"train {r['train_s']:7.1f}s  rows {len(Xt)}", flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"screen_fold{fold}.json").write_text(json.dumps(res, indent=1))
    return res


def summary(folds):
    rows = {}
    for f in folds:
        p = RES / f"screen_fold{f}.json"
        if not p.exists():
            continue
        for a, d in json.loads(p.read_text()).items():
            for mode, r in d.items():
                rows.setdefault((a, mode), []).append(r)
    out = []
    for (a, mode), rs in rows.items():
        out.append({"algo": a, "mode": mode, "folds": len(rs), **{k: round(float(np.mean([r[k] for r in rs])), 3)
                                                                   for k in ("val_full_m", "val_degraded_m", "criterion", "within_5m", "train_s")}})
    return sorted(out, key=lambda r: (r["mode"], r["criterion"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,3,6")
    ap.add_argument("--algos", default=",".join(ALL))
    a = ap.parse_args()
    for f in [int(x) for x in a.folds.split(",")]:
        screen(f, a.algos.split(","))
    print(json.dumps(summary([int(x) for x in a.folds.split(",")]), indent=1))
