"""Fit horizon-aware uncertainty calibration (sigma deciles within time-unseen buckets) for a recon model.

Uses only the time-blocked validation tail of each fold's TRAINING matches (the same rows the global calibration was
fitted on), never the test match. The table is stored as `calibration_h` beside the global `calibration` in each
checkpoint (a backup of the original file is kept as *.pre_horizon.pt); HiddenModel.calibrate uses it when present.

    .venv/bin/python -m backend.reconstruction.calibrate_horizon --tag r13_mlp_aug_ens3
"""
import argparse
import shutil

import numpy as np

from backend.reconstruction import hidden_models as hm
from backend.reconstruction.ml import calibrate
from backend.reconstruction.train_hidden import OUT, VAL_FROM_T, _sel, assemble, fold_split

MIN_ROWS = 2000


def fit(tag, fold):
    import torch
    train_ms, test = fold_split(fold)
    out = {}
    for mode in ("live", "replay"):
        path = OUT / tag / f"hidden_{mode}_fold{fold}.pt"
        model = hm.load_hidden(path)
        first = model.members[0].ck if model.arch == "ensemble" else model.ck
        d = assemble(train_ms, mode, augment=bool(first.get("augment")))
        va = _sel(d, d["tabs"] >= VAL_FROM_T)
        seq = va["store"].windows(va["inst"], va["t"], va["j"]) if model.needs_context else None
        mu, sig = model.raw(va["X"], seq)
        err = np.linalg.norm(mu - va["Y"] * 20, axis=1)
        b = hm.horizon_bucket(va["X"])
        tabs = {name: calibrate(sig[b == name], err[b == name]) for name in np.unique(b) if (b == name).sum() >= MIN_ROWS}
        ck = torch.load(path, weights_only=False)
        backup = path.with_suffix(".pre_horizon.pt")
        if not backup.exists():
            shutil.copy2(path, backup)
        ck["calibration_h"] = tabs
        ck["calibration_h_rows"] = {name: int((b == name).sum()) for name in np.unique(b)}
        torch.save(ck, path)
        # in-sample check on the fitting rows (the honest check is the held-out benchmark)
        m2 = hm.HiddenModel(ck, model.members) if model.arch == "ensemble" else hm.load_hidden(path)
        r80, _ = m2.calibrate(sig, va["X"])
        r80g, _ = model.calibrate(sig, va["X"])
        out[mode] = {name: (round(float((err[b == name] <= r80g[b == name]).mean()), 3), round(float((err[b == name] <= r80[b == name]).mean()), 3))
                     for name in np.unique(b)}
        print(f"fold {fold} {mode}: coverage global -> horizon (fit rows) {out[mode]}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--folds", default="0,1,2,3,4,5,6,final")
    a = ap.parse_args()
    for f in a.folds.split(","):
        fit(a.tag, f if f == "final" else int(f))
