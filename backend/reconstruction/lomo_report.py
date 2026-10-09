"""Aggregate leave-one-match-out results (backend/reconstruction/ml.py --lomo) into one table.

Each of the 7 DFL matches is the held-out test match once (validation = next match, train = other 5).
Reports per mask and method: mean / sd / min / max over folds of hidden-player MAE, within 5 m / 10 m, and the
80% radius coverage for the learned models.

    .venv/bin/python -m backend.reconstruction.lomo_report            # v1 features
    .venv/bin/python -m backend.reconstruction.lomo_report --feat 2   # v2 features; --compare for v1 vs v2
"""
import argparse
import glob
import json

import numpy as np

from backend.reconstruction.datasets import CACHE


def load(feat=1):
    suffix = "" if feat == 1 else f"_v{feat}"
    out = []
    for i in range(7):
        f = CACHE / f"ml_results_lomo{i}{suffix}.json"
        if f.exists():
            out.append(json.load(open(f)))
    return out


def load_pattern(pattern):
    """Folds saved as ml_results_<pattern.format(i)>.json, e.g. "k7r_{}_v3" (7-fold: 6 train matches, refit)."""
    out = []
    for i in range(7):
        f = CACHE / f"ml_results_{pattern.format(i)}.json"
        if f.exists():
            out.append(json.load(open(f)))
    return out


def aggregate(folds):
    res = {}
    for mk in folds[0]["results"]:
        res[mk] = {}
        for method in folds[0]["results"][mk]:
            vals = [f["results"][mk][method] for f in folds if method in f["results"].get(mk, {})]
            row = {}
            for k in ("mae_m", "rmse_m", "within_5m", "within_10m", "radius_coverage_80"):
                v = [x[k] for x in vals if x.get(k) is not None]
                if v:
                    row[k] = {"mean": round(float(np.mean(v)), 3), "sd": round(float(np.std(v)), 3),
                              "min": round(float(np.min(v)), 3), "max": round(float(np.max(v)), 3)}
            res[mk][method] = row
    return {"folds": len(folds), "test_matches": [f["test_match"] for f in folds], "results": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", type=int, default=1)
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    feats = (1, 2) if a.compare else (a.feat,)
    report = {}
    for fv in feats:
        folds = load(fv)
        if not folds:
            print(f"no folds for feature version {fv}")
            continue
        agg = aggregate(folds)
        report[f"v{fv}"] = agg
        print(f"\nfeature version {fv}: {agg['folds']} folds")
        for mk in ("sc_camera", "k=18", "k=14", "k=10", "k=6"):
            if mk not in agg["results"]:
                continue
            r = agg["results"][mk]
            print(f"  {mk:10s} " + " | ".join(f"{m} {v['mae_m']['mean']:.2f}±{v['mae_m']['sd']:.2f}" for m, v in r.items()))
            for m in ("ml_live", "ml_replay"):
                if "radius_coverage_80" in r.get(m, {}):
                    print(f"  {'':10s} {m} coverage80 {r[m]['radius_coverage_80']['mean']:.3f} (min {r[m]['radius_coverage_80']['min']:.3f})")
    # 7-fold recipes (train on 6 matches, test on the 7th)
    for key, pat in (("k7_v2", "k7_{}_v2"), ("k7r_v2", "k7r_{}_v2"), ("k7r_v3", "k7r_{}_v3")):
        folds = load_pattern(pat)
        if len(folds) == 7:
            report[key] = aggregate(folds)
            r = report[key]["results"]["sc_camera"]
            print(f"{key}: live {r['ml_live']['mae_m']['mean']:.2f}±{r['ml_live']['mae_m']['sd']:.2f}  replay {r['ml_replay']['mae_m']['mean']:.2f}±{r['ml_replay']['mae_m']['sd']:.2f}")
    json.dump(report, open(CACHE / "lomo_report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
