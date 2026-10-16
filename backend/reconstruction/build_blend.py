"""Blend of two trained ensembles (equal weights): the production 3-seed MLP ensemble and the algo_zoo top-3 ensemble
(MLP + mixture density network + zone classifier). Calibrated on the same time-blocked validation tail; held-out
checks are preserved (both members record the same train / test matches).

    .venv/bin/python -m backend.reconstruction.build_blend --a r13_mlp_aug_ens3 --b r16_top3 --tag r16_blend
"""
import argparse

import numpy as np

from backend.reconstruction import hidden_models as hm
from backend.reconstruction.ml import calibrate
from backend.reconstruction.train_hidden import OUT, VAL_FROM_T, _sel, assemble, fold_split


def build(a, b, tag, fold):
    import torch
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)
    train_ms, test = fold_split(fold)
    res = {}
    for mode in ("live", "replay"):
        pa, pb = (OUT / a / f"hidden_{mode}_fold{fold}.pt").resolve(), (OUT / b / f"hidden_{mode}_fold{fold}.pt").resolve()
        ma, mb = hm.load_hidden(pa), hm.load_hidden(pb)
        for m in (ma, mb):
            assert m.ck.get("test") == test and sorted(m.ck.get("train") or []) == sorted(train_ms)
        rel = [f"../{a}/hidden_{mode}_fold{fold}.pt", f"../{b}/hidden_{mode}_fold{fold}.pt"]  # members resolve per file
        ck = {"arch": "ensemble", "members": rel, "mode": mode, "fold": fold, "train": train_ms, "test": test,
              "feat_version": 3, "calibration": None, "blend_of": [a, b]}
        model = hm.HiddenModel(ck, [ma, mb])
        d = assemble(train_ms, mode, augment=True)
        va = _sel(d, d["tabs"] >= VAL_FROM_T)
        mu, sig = model.raw(va["X"])
        err = np.linalg.norm(mu - va["Y"] * 20, axis=1)
        ck["calibration"] = calibrate(sig, err)
        bk = hm.horizon_bucket(va["X"])
        ck["calibration_h"] = {n: calibrate(sig[bk == n], err[bk == n]) for n in np.unique(bk) if (bk == n).sum() >= 2000}
        sc, dg = va["mask"] == "sc_camera", va["mask"] == "degraded"
        ck["val"] = {"val_mae_sc_camera": float(err[sc].mean()), "val_mae_degraded": float(err[dg].mean())}
        torch.save(ck, out / f"hidden_{mode}_fold{fold}.pt")
        res[mode] = ck["val"]
        print(fold, mode, {k: round(v, 3) for k, v in ck["val"].items()}, flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--folds", default="0,1,2,3,4,5,6,final")
    x = ap.parse_args()
    for f in x.folds.split(","):
        build(x.a, x.b, x.tag, f if f == "final" else int(f))
