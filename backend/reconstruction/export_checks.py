"""Quality checks for reconstructed match exports (Phase 9).

Per file: timestamp continuity, coordinate bounds, players on pitch per second, measured positions unchanged vs a
reference (when given), physically implausible steps of ESTIMATED positions, radius sanity, live/replay presence.

    .venv/bin/python -m backend.reconstruction.export_checks data/skc_*_h1.json ...
"""
import json
import sys
from pathlib import Path

import numpy as np

VMAX_EST = 10.0  # m/s between consecutive seconds for an estimated position (sprint ~ 9-10 m/s)


def _xy(frames, N):
    return np.array([[p if p is not None else [np.nan, np.nan] for p in f["xy"]] for f in frames], float).reshape(len(frames), N, 2)


def check(path):
    d = json.loads(Path(path).read_text())
    F = d["frames"]
    N = len(d["meta"]["player_order"])
    out = {"file": str(path), "seconds": len(F), "players": N, "issues": []}
    ts = np.array([f["t"] for f in F])
    gaps = np.diff(ts)
    out["timestamp_continuous"] = bool((gaps == 1).all())
    if not out["timestamp_continuous"]:
        out["issues"].append(f"non-unit timestamp steps: {int((gaps != 1).sum())}")
    views = {"frames": F, **{f"alt_{k}": v for k, v in (d.get("alt_frames") or {}).items()}}
    for name, fr in views.items():
        xy = _xy(fr, N)
        fin = np.isfinite(xy).all(2)
        oob = fin & ((xy[..., 0] < -3) | (xy[..., 0] > 108) | (xy[..., 1] < -3) | (xy[..., 1] > 71))
        on = fin.sum(1)
        obs = np.array([[o is True for o in f.get("obs", [True] * N)] for f in fr])
        est = fin & ~obs
        step = np.linalg.norm(np.diff(xy, axis=0), axis=2)
        both_est = est[1:] & est[:-1]
        fast = both_est & (step > VMAX_EST)
        rad = np.array([[r if r is not None else np.nan for r in f.get("rad", [None] * N)] for f in fr], float)
        r = {"out_of_bounds": int(oob.sum()), "out_of_bounds_estimated": int((oob & est).sum()),
             "seconds_without_tracking": int((on == 0).sum()), "players_on_min": int(on.min()), "players_on_max": int(on.max()),
             "seconds_with_more_than_22": int((on > 22).sum()), "estimated_share": round(float(est.sum() / max(1, fin.sum())), 3),
             "estimated_steps_over_10ms": int(fast.sum()), "estimated_steps": int(both_est.sum()),
             "estimated_without_radius": int((est & ~np.isfinite(rad)).sum()) if "rad" in fr[0] else None}
        out[name] = r
        if r["out_of_bounds_estimated"]:
            out["issues"].append(f"{name}: {r['out_of_bounds_estimated']} ESTIMATED positions out of bounds")
        if r["out_of_bounds"] - r["out_of_bounds_estimated"]:
            out.setdefault("notes", []).append(f"{name}: {r['out_of_bounds'] - r['out_of_bounds_estimated']} measured / ground-truth "
                                               "positions > 3 m outside the pitch (kept as measured)")
        if r["seconds_with_more_than_22"]:
            out["issues"].append(f"{name}: {r['seconds_with_more_than_22']} seconds with more than 22 players")
        if r["estimated_steps_over_10ms"]:
            out["issues"].append(f"{name}: {r['estimated_steps_over_10ms']} estimated steps > {VMAX_EST} m/s")
    if "alt_replay" in out and "alt_frames" in d and d["meta"].get("view") != "truth":
        # measured positions identical in live and replay
        a, b = _xy(d["alt_frames"]["replay"], N), _xy(d["alt_frames"]["live"], N)
        oa = np.array([[o is True for o in f["obs"]] for f in d["alt_frames"]["replay"]])
        out["measured_identical_live_replay"] = bool(np.allclose(a[oa], b[oa], equal_nan=True))
    out["ok"] = not out["issues"]
    return out


if __name__ == "__main__":
    res = [check(p) for p in sys.argv[1:]]
    print(json.dumps(res, indent=1))
