"""Evaluate hidden-state reconstruction and tactical detection against synthetic ground truth.

    .venv/bin/python -m backend.reconstruction.evaluate            # calibrate (seed 11) + evaluate (seed 7)

Calibration (uncertainty radius / confidence per time-since-sighting bucket) is fitted on a different
synthetic match (seed 11) than the one evaluated (seed 7). All numbers are computed, none hand-written.
"""
import json
import time
from pathlib import Path

import numpy as np

from backend.analytics.possession import possession_sequences
from backend.models.match import L, MatchData, to_own
from backend.reconstruction.estimator import CALIB, DT_BUCKETS, N_KEYS, SRC_INF, reconstruct, roster_timeline
from backend.reconstruction.observation import observe
from backend.tactics.counterattack import detect_counterattacks
from backend.tactics.shift import detect_shifts
from backend.tactics.tactical_state import build_timeline, tactical_label
from backend.tactics.transitions import detect_transitions
from simulator.match_generator import generate_match

OUT = CALIB.parent
LEVELS = ["k=18", "k=14", "k=10", "k=6", "broadcast"]
EVAL_SEED, CALIB_SEED = 7, 11


# ----------------------------------------------------------------------------- reconstruction
def calibrate(seed=CALIB_SEED):
    m = MatchData(generate_match(seed))
    table = {}
    for mode in ("live", "replay"):
        errs = {k: [] for k in range(N_KEYS)}
        for lvl in LEVELS:
            obs = observe(m, lvl, seed)
            rec = reconstruct(m, obs.vis, mode, calib={})
            hid = rec.src == SRC_INF
            e = np.linalg.norm(rec.pos - m.xy, axis=2)
            for k in errs:
                sel = hid & (rec.key == k) & np.isfinite(e)
                errs[k] += list(e[sel])
        stat = lambda v: {"r80": round(float(np.percentile(v, 80)), 2) if v else 15.0,
                          "p5": round(float(np.mean(np.array(v) <= 5)), 3) if v else 0.2, "n": len(v)}
        table[mode] = {str(k): stat(v) for k, v in errs.items()}
        for d in range(len(DT_BUCKETS) + 1):  # time-only pools as fallback for sparse cells
            table[mode][f"d{d}"] = stat(sum((errs[d * 3 + v] for v in range(3)), []))
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(table, open(CALIB, "w"), indent=1)
    return table


def shape_metrics(pos, team, mask):
    """Team shape from a full-state array (T, n, 2): def line, width, depth, centroid x."""
    p = to_own(team, pos[..., 0], pos[..., 1])
    x, y = np.where(mask, p[0], np.nan), np.where(mask, p[1], np.nan)
    with np.errstate(all="ignore"):
        xs = np.sort(x, axis=1)
        return {"def_line": np.nanmean(xs[:, :4], 1), "width": np.nanmax(y, 1) - np.nanmin(y, 1),
                "depth": np.nanmax(x, 1) - np.nanmin(x, 1), "centroid_x": np.nanmean(x, 1)}


def hold_last_baseline(m, vis, on):
    pos = np.full_like(m.xy, np.nan)
    last = np.full(m.xy.shape[1:], np.nan)
    for t in range(m.T):
        last = np.where(vis[t][:, None], m.xy[t], last)
        pos[t] = last
    return pos


def score_reconstruction(m, rec_pos, vis, on, rad=None, conf=None):
    truth = m.xy
    hid = on & ~vis & ~np.isnan(truth[..., 0]) & ~np.isnan(rec_pos[..., 0])
    e = np.linalg.norm(rec_pos - truth, axis=2)[hid]
    out = {"hidden_samples": int(hid.sum()), "mae_m": round(float(e.mean()), 2), "rmse_m": round(float(np.sqrt((e ** 2).mean())), 2),
           "within_5m": round(float((e <= 5).mean()), 3), "within_10m": round(float((e <= 10).mean()), 3)}
    if rad is not None:
        out["radius_coverage (target 0.80)"] = round(float((e <= rad[hid]).mean()), 3)
        c = conf[hid]
        hit = (e <= 5).astype(float)
        bins = [(0, .25), (.25, .5), (.5, .75), (.75, 1.01)]
        out["confidence_reliability"] = [{"conf_bin": f"{a:.2f}-{b:.2f}", "mean_conf": round(float(c[(c >= a) & (c < b)].mean()), 3),
                                          "actual_within_5m": round(float(hit[(c >= a) & (c < b)].mean()), 3), "n": int(((c >= a) & (c < b)).sum())}
                                         for a, b in bins if ((c >= a) & (c < b)).any()]
    # full-team shape error (all on-pitch outfield players, observed + reconstructed)
    out_field = np.array([not (p.get("start_role") == "GK" or p.get("position") == "GK") for p in m.players])
    shape = {}
    for team in ("A", "B"):
        mask = on & np.array([pid.startswith(team) for pid in m.order])[None] & out_field[None]
        st = shape_metrics(np.where(mask[..., None], rec_pos, np.nan), team, mask)
        gt = shape_metrics(np.where(mask[..., None], truth, np.nan), team, mask)
        for k in st:
            d = np.abs(st[k] - gt[k])
            shape.setdefault(k, []).append(d[np.isfinite(d)])
    out["team_shape_mae_m"] = {k: round(float(np.concatenate(v).mean()), 2) for k, v in shape.items()}
    return out


# ----------------------------------------------------------------------------- tactics
def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0,
            "tp": tp, "fp": fp, "fn": fn}


def match_events(pred, gt, tol, key=lambda x: (x["team"], x["t"])):
    used, tp = set(), 0
    for p in pred:
        pt, ptt = key(p)
        best = None
        for i, g in enumerate(gt):
            gt_team, gtt = key(g)
            if i not in used and gt_team == pt and abs(gtt - ptt) <= tol and (best is None or abs(gtt - ptt) < abs(key(gt[best])[1] - ptt)):
                best = i
        if best is not None:
            used.add(best)
            tp += 1
    return prf(tp, len(pred) - tp, len(gt) - tp)


def tactical_scores(m: MatchData, truth: dict):
    tl = build_timeline(m)
    seqs = possession_sequences(m)
    res = {}
    # HIGH_PRESS per team-minute
    tp = fp = fn = 0
    for team, opp in (("A", "B"), ("B", "A")):
        for row, orow, g in zip(tl[team], tl[opp], truth["tactic_by_minute"][team]):
            pred = tactical_label(row, orow) == "HIGH_PRESS"
            tp += pred and g["high_press"]
            fp += pred and not g["high_press"]
            fn += (not pred) and g["high_press"]
    res["HIGH_PRESS (team-minutes)"] = prf(tp, fp, fn)
    # TACTICAL_SHIFT within +-3 min of the simulator's tactic changes
    shifts = [{"team": s["team"], "t": s["minute"] * 60} for s in detect_shifts(tl)]
    res["TACTICAL_SHIFT (+-3 min)"] = match_events(shifts, truth["tactic_changes"], 180)
    # COUNTERATTACK vs the simulator's counter-mode activations (+-5 s)
    ca = [{"team": c["team"], "t": c["start"]} for c in detect_counterattacks(m, seqs)]
    res["COUNTERATTACK (+-5 s)"] = match_events(ca, [c for c in truth["counter_starts"] if c.get("developed")], 5)
    # transitions: a recovery the team keeps for >= 5 s is a genuine transition (simulator ground truth)
    rec = truth["recoveries"]
    gt_tr = [r for i, r in enumerate(rec) if (rec[i + 1]["t"] if i + 1 < len(rec) else 1e9) - r["t"] >= 5]
    att, dfn = detect_transitions(m)
    res["ATTACKING_TRANSITION (+-3 s)"] = match_events(att, gt_tr, 3)
    res["DEFENSIVE_TRANSITION (+-3 s)"] = match_events(dfn, [{"team": "B" if r["team"] == "A" else "A", "t": r["t"]} for r in gt_tr], 3)
    return res


def variant(m_truth: MatchData, xy):
    m = MatchData.__new__(MatchData)
    m.__dict__.update(m_truth.__dict__)
    m.xy = xy
    return m


def run(levels=LEVELS):
    t0 = time.time()
    calib = calibrate()
    raw = generate_match(EVAL_SEED)
    m = MatchData(raw)
    truth = raw["truth"]
    on, _ = roster_timeline(m)
    report = {"eval_seed": EVAL_SEED, "calibration_seed": CALIB_SEED, "reconstruction": {}, "tactics": {}}
    for lvl in levels:
        obs = observe(m, lvl, EVAL_SEED)
        vis = obs.vis & on
        row = {"observed_share": round(float(vis.sum() / on.sum()), 3),
               "avg_players_observed": round(float(vis.sum(1).mean()), 1)}
        row["hold_last_seen_baseline"] = score_reconstruction(m, hold_last_baseline(m, vis, on), vis, on)
        recs = {}
        for mode in ("live", "replay"):
            rec = reconstruct(m, vis, mode, calib)
            recs[mode] = rec
            row[mode] = score_reconstruction(m, rec.pos, vis, on, rec.rad, rec.conf)
        report["reconstruction"][lvl] = row
        if lvl in ("broadcast", "k=10", "k=6"):
            observed_only = np.where(vis[..., None], m.xy, np.nan)
            report["tactics"][lvl] = {
                "observed_only": tactical_scores(variant(m, observed_only), truth),
                "reconstructed_live": tactical_scores(variant(m, recs["live"].pos), truth),
                "reconstructed_replay": tactical_scores(variant(m, recs["replay"].pos), truth),
            }
        print(f"{lvl}: done ({time.time() - t0:.0f}s)", flush=True)
    report["tactics"]["full_ground_truth"] = tactical_scores(m, truth)
    report["seconds"] = round(time.time() - t0, 1)
    json.dump(report, open(OUT / "report.json", "w"), indent=1)
    return report


def summary(rep):
    lines = ["RECONSTRUCTION (hidden players only; seed %d, calibrated on seed %d)" % (rep["eval_seed"], rep["calibration_seed"]),
             f"{'visibility':>10} {'obs/22':>6} | {'hold-last MAE':>13} | {'live MAE':>8} {'<5m':>5} {'<10m':>5} {'cov80':>5} | {'replay MAE':>10} {'<5m':>5} {'<10m':>5} {'cov80':>5} | {'shape(def line) live/replay':>27}"]
    for lvl, r in rep["reconstruction"].items():
        lv, rp = r["live"], r["replay"]
        lines.append(f"{lvl:>10} {r['avg_players_observed']:>6} | {r['hold_last_seen_baseline']['mae_m']:>13} | {lv['mae_m']:>8} {lv['within_5m']:>5} {lv['within_10m']:>5} "
                     f"{lv['radius_coverage (target 0.80)']:>5} | {rp['mae_m']:>10} {rp['within_5m']:>5} {rp['within_10m']:>5} {rp['radius_coverage (target 0.80)']:>5} | "
                     f"{lv['team_shape_mae_m']['def_line']:>12}/{rp['team_shape_mae_m']['def_line']}")
    lines.append("\nTACTICS (F1)")
    keys = list(rep["tactics"]["full_ground_truth"])
    lines.append(f"{'state':>34} | " + " | ".join(f"{k.split(' ')[0][:14]:>14}" for k in keys))
    lines.append(f"{'full ground truth':>34} | " + " | ".join(f"{rep['tactics']['full_ground_truth'][k]['f1']:>14}" for k in keys))
    for lvl, v in rep["tactics"].items():
        if lvl == "full_ground_truth":
            continue
        for name, sc in v.items():
            lines.append(f"{lvl + ' ' + name:>34} | " + " | ".join(f"{sc[k]['f1']:>14}" for k in keys))
    return "\n".join(lines)


if __name__ == "__main__":
    rep = run()
    print(summary(rep))
