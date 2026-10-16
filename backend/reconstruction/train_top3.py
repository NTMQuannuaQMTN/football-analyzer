"""Train the top-3 algorithms from the screening (algo_zoo: MLP, mixture density network, zone classifier) as one
ensemble, per 7-fold fold (test match never read) and on all 7 matches ('final').

Same rows and recipe as the screening (v3 features, context-degradation augmentation, 10 epochs each); the ensemble
mean is the correction, sigma combines the probabilistic members' variances with the members' spread, and the
radius / P(err <= 5 m) are calibrated on the time-blocked validation tail (global sigma deciles + time-unseen buckets).

    .venv/bin/python -m backend.reconstruction.train_top3 --folds 0,1,2,3,4,5,6,final --tag r16_top3
"""
import argparse
import json
import time

import numpy as np

from backend.reconstruction import hidden_models as hm
from backend.reconstruction.algo_zoo import SEED, _torch_fit
from backend.reconstruction.ml import calibrate
from backend.reconstruction.train_hidden import FEAT, OUT, VAL_FROM_T, VAL_GAP_S, _sel, _source_digest, assemble, fold_split

MIN_H_ROWS = 2000


def _train_member(kind, X, Y):
    import torch
    from torch import nn
    n_in = X.shape[1]
    if kind == "mlp":
        loss = lambda n, x, y: nn.functional.huber_loss(n(x), y, delta=0.25)
        return _torch_fit(lambda: hm.make_zoo_net("mlp", n_in), loss, X, Y)
    if kind == "mdn":
        K = hm.ZOO_MDN_K

        def loss(n, x, y):
            o = n(x).view(-1, K, 5)
            logit, mu, ls = o[..., 0], o[..., 1:3], o[..., 3:5].clamp(-6, 3)
            z = (y[:, None, :] - mu) / torch.exp(ls)
            logp = -0.5 * (z ** 2).sum(-1) - ls.sum(-1) + torch.log_softmax(logit, -1)
            return -torch.logsumexp(logp, -1).mean()
        return _torch_fit(lambda: hm.make_zoo_net("mdn", n_in), loss, X, Y)
    nb = int(2 * hm.ZOO_ZONE_R / hm.ZOO_ZONE_M)
    cells = np.clip(((Y * 20 + hm.ZOO_ZONE_R) / hm.ZOO_ZONE_M).astype(int), 0, nb - 1)
    lab = (cells[:, 0] * nb + cells[:, 1]).astype(np.int64)
    return _torch_fit(lambda: hm.make_zoo_net("zone", n_in), lambda n, x, y: nn.functional.cross_entropy(n(x), y), X, lab)


def train_fold(fold, tag):
    import torch
    train_ms, test = fold_split(fold)
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)
    summary = {"fold": fold, "train": train_ms, "test": test, "modes": {}}
    for mode in ("live", "replay"):
        t0 = time.time()
        d = assemble(train_ms, mode, augment=True)
        tr = _sel(d, d["tabs"] < VAL_FROM_T - VAL_GAP_S)
        va = _sel(d, d["tabs"] >= VAL_FROM_T)
        members = {kind: _train_member(kind, tr["X"], tr["Y"]) for kind in ("mlp", "mdn", "zone")}
        ck = {"arch": "zoo_top3", "members": {k: n.state_dict() for k, n in members.items()}, "n_in": tr["X"].shape[1],
              "mode": mode, "fold": fold, "train": train_ms, "test": test, "feat_version": FEAT, "calibration": None,
              "augment": "degraded broadcast context (30 s clips, no possession/status/roles)", "seed": SEED,
              "protocol": "7-fold match-level; members trained 10 epochs on rows before the validation tail; calibration "
                          "on the time-blocked tail of the training halves", "source_digest": _source_digest(),
              "selected_by": "algo_zoo screening (validation tail of training matches, folds 0/3/6): MLP, MDN, zone classifier"}
        model = hm.HiddenModel(ck)
        mu, sig = model.raw(va["X"])
        err = np.linalg.norm(mu - va["Y"] * 20, axis=1)
        ck["calibration"] = calibrate(sig, err)
        b = hm.horizon_bucket(va["X"])
        ck["calibration_h"] = {name: calibrate(sig[b == name], err[b == name]) for name in np.unique(b) if (b == name).sum() >= MIN_H_ROWS}
        sc, dg = va["mask"] == "sc_camera", va["mask"] == "degraded"
        ck["val"] = {"val_mae_sc_camera": float(err[sc].mean()), "val_mae_degraded": float(err[dg].mean()), "val_mae_all": float(err.mean())}
        torch.save(ck, out / f"hidden_{mode}_fold{fold}.pt")
        summary["modes"][mode] = {**ck["val"], "seconds": round(time.time() - t0)}
        print(f"[{time.strftime('%H:%M:%S')}] fold {fold} {mode}: val full {ck['val']['val_mae_sc_camera']:.3f} "
              f"degraded {ck['val']['val_mae_degraded']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    (out / f"summary_fold{fold}.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4,5,6,final")
    ap.add_argument("--tag", default="r16_top3")
    a = ap.parse_args()
    for f in a.folds.split(","):
        train_fold(f if f == "final" else int(f), a.tag)
