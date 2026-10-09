"""Evaluate reconstruction + tactical detection over many synthetic matches (seeds), in parallel.

    .venv/bin/python -m backend.reconstruction.multi_eval --matches 100 --workers 6

Uses the fixed calibration from seed 11 (excluded from the evaluation seeds). Every match follows the
same tactical script with different random play, so the spread measures match-to-match variation.
Output: data/eval/multi_report.json (per-match rows + mean / std / 95% CI / min / max).
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from backend.reconstruction.evaluate import CALIB_SEED, LEVELS, OUT  # noqa: E402

TACTIC_LEVELS = ("broadcast", "k=6")


def run_one(seed: int) -> dict:
    from backend.models.match import MatchData
    from backend.reconstruction.estimator import load_calibration, reconstruct, roster_timeline
    from backend.reconstruction.evaluate import hold_last_baseline, score_reconstruction, tactical_scores, variant
    from backend.reconstruction.observation import observe
    from simulator.match_generator import generate_match

    t0 = time.time()
    calib = load_calibration()
    raw = generate_match(seed)
    m = MatchData(raw)
    truth = raw["truth"]
    on, _ = roster_timeline(m)
    row = {"seed": seed, "score": raw["meta"]["final_score"], "recon": {}, "tactics": {}}
    for lvl in LEVELS:
        vis = observe(m, lvl, seed).vis & on
        r = {"avg_observed": float(vis.sum(1).mean()),
             "hold": score_reconstruction(m, hold_last_baseline(m, vis, on), vis, on)}
        recs = {}
        for mode in ("live", "replay"):
            rec = reconstruct(m, vis, mode, calib)
            recs[mode] = rec
            s = score_reconstruction(m, rec.pos, vis, on, rec.rad, rec.conf)
            s.pop("confidence_reliability", None)
            r[mode] = s
        row["recon"][lvl] = r
        if lvl in TACTIC_LEVELS:
            row["tactics"][lvl] = {
                "observed_only": tactical_scores(variant(m, np.where(vis[..., None], m.xy, np.nan)), truth),
                "reconstructed_replay": tactical_scores(variant(m, recs["replay"].pos), truth),
            }
    row["tactics"]["full_ground_truth"] = tactical_scores(m, truth)
    row["seconds"] = round(time.time() - t0, 1)
    return row


def stats(v):
    v = np.array([x for x in v if x is not None and np.isfinite(x)], dtype=float)
    if not len(v):
        return None
    sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    half = 1.96 * sd / np.sqrt(len(v))
    return {"mean": round(float(v.mean()), 3), "std": round(sd, 3), "ci95": [round(float(v.mean() - half), 3), round(float(v.mean() + half), 3)],
            "min": round(float(v.min()), 3), "max": round(float(v.max()), 3), "n": len(v)}


def aggregate(rows):
    agg = {"recon": {}, "tactics": {}}
    for lvl in LEVELS:
        a = {"avg_observed": stats([r["recon"][lvl]["avg_observed"] for r in rows])}
        for mode in ("hold", "live", "replay"):
            a[mode] = {k: stats([r["recon"][lvl][mode][k] for r in rows])
                       for k in ("mae_m", "rmse_m", "within_5m", "within_10m")}
            if mode != "hold":
                a[mode]["radius_coverage"] = stats([r["recon"][lvl][mode]["radius_coverage (target 0.80)"] for r in rows])
                a[mode]["def_line_mae_m"] = stats([r["recon"][lvl][mode]["team_shape_mae_m"]["def_line"] for r in rows])
        a["replay_beats_hold_share"] = round(float(np.mean([r["recon"][lvl]["replay"]["mae_m"] < r["recon"][lvl]["hold"]["mae_m"] for r in rows])), 3)
        a["live_beats_hold_share"] = round(float(np.mean([r["recon"][lvl]["live"]["mae_m"] < r["recon"][lvl]["hold"]["mae_m"] for r in rows])), 3)
        agg["recon"][lvl] = a
    keys = list(rows[0]["tactics"]["full_ground_truth"])
    for state in ["full_ground_truth"] + [f"{l}|{v}" for l in TACTIC_LEVELS for v in ("observed_only", "reconstructed_replay")]:
        get = (lambda r, k: r["tactics"]["full_ground_truth"][k]) if state == "full_ground_truth" else \
              (lambda r, k, s=state: r["tactics"][s.split("|")[0]][s.split("|")[1]][k])
        agg["tactics"][state] = {}
        for k in keys:
            tp = sum(get(r, k)["tp"] for r in rows)
            fp = sum(get(r, k)["fp"] for r in rows)
            fn = sum(get(r, k)["fn"] for r in rows)
            p = tp / (tp + fp) if tp + fp else 0.0
            rc = tp / (tp + fn) if tp + fn else 0.0
            agg["tactics"][state][k] = {"f1_per_match": stats([get(r, k)["f1"] for r in rows]),
                                        "pooled": {"precision": round(p, 3), "recall": round(rc, 3),
                                                   "f1": round(2 * p * rc / (p + rc), 3) if p + rc else 0.0, "tp": tp, "fp": fp, "fn": fn}}
    return agg


def main(n, workers):
    seeds = [s for s in range(1, n + 2) if s != CALIB_SEED][:n]
    rows, failed = [], []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_one, s): s for s in seeds}
        for i, f in enumerate(as_completed(futs), 1):
            s = futs[f]
            try:
                rows.append(f.result())
            except Exception as e:  # keep going; report failures
                failed.append({"seed": s, "error": repr(e)[:300]})
            if i % 5 == 0 or i == len(seeds):
                print(f"{i}/{len(seeds)} matches done ({time.time() - t0:.0f}s, failed {len(failed)})", flush=True)
    rows.sort(key=lambda r: r["seed"])
    rep = {"matches": len(rows), "seeds": [r["seed"] for r in rows], "failed": failed, "calibration_seed": CALIB_SEED,
           "note": "same tactical script, different random play per seed; calibration fitted on seed 11 only",
           "seconds": round(time.time() - t0, 1), "aggregate": aggregate(rows), "per_match": rows}
    json.dump(rep, open(OUT / "multi_report.json", "w"), indent=1)
    return rep


def summary(rep):
    a = rep["aggregate"]
    f = lambda s: f"{s['mean']:.2f} ±{(s['ci95'][1] - s['ci95'][0]) / 2:.2f}"
    out = [f"{rep['matches']} matches (failed {len(rep['failed'])}), {rep['seconds']:.0f}s",
           "HIDDEN-PLAYER ERROR, mean over matches ±95% CI (m)",
           f"{'visibility':>10} {'obs':>5} | {'freeze':>11} | {'live':>11} | {'replay':>11} | {'replay <=5m':>11} {'<=10m':>7} | {'radius hit':>10} | replay beats freeze"]
    for lvl, r in a["recon"].items():
        out.append(f"{lvl:>10} {r['avg_observed']['mean']:>5.1f} | {f(r['hold']['mae_m']):>11} | {f(r['live']['mae_m']):>11} | {f(r['replay']['mae_m']):>11} | "
                   f"{r['replay']['within_5m']['mean']:>11.1%} {r['replay']['within_10m']['mean']:>7.1%} | {r['replay']['radius_coverage']['mean']:>10.1%} | "
                   f"{r['replay_beats_hold_share']:.0%} of matches")
    out.append("\nTACTICAL F1 (pooled over all matches; per-match mean ±95% CI in brackets)")
    keys = list(a["tactics"]["full_ground_truth"])
    for state, d in a["tactics"].items():
        out.append(f"{state:>34}: " + " | ".join(f"{k.split(' ')[0][:14]} {d[k]['pooled']['f1']:.3f} [{f(d[k]['f1_per_match'])}]" for k in keys))
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=100)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    print(summary(main(a.matches, a.workers)))
