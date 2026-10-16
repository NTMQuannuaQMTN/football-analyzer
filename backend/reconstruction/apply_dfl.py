"""Re-export the 7 DFL matches (14 halves) with a recon model, each half using the fold that held its match out.

The match frames stay ground truth (analysis runs on it); `alt_frames` (camera replay / live) are rebuilt with the
same camera footprint seed as the original exports (seed 0) so the files are directly comparable. Event pins are OFF:
the pin clock lag is fitted over the whole half, which would give the live view future information. Writes
data/dfl_<code>_<tag>_h<half>.json beside the originals (never overwritten) and a metrics log.

    .venv/bin/python -m backend.reconstruction.apply_dfl --model r13_mlp_ens3 --tag r13
--file-tag keeps an existing export's file name (and app id) while recording the new session, e.g.
    .venv/bin/python -m backend.reconstruction.apply_dfl --model r16_blend --tag r16 --file-tag r13
"""
import argparse
import json
from pathlib import Path

import numpy as np

from backend.reconstruction import dfl_match
from backend.reconstruction.datasets import dfl_matches

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def hidden_metrics(data, mode):
    truth = np.array([[p if p is not None else [np.nan, np.nan] for p in f["xy"]] for f in data["frames"]], float)
    alt = data["alt_frames"][mode]
    pred = np.array([[p if p is not None else [np.nan, np.nan] for p in f["xy"]] for f in alt], float)
    obs = np.array([[o is True for o in f["obs"]] for f in alt])
    ok = np.isfinite(truth).all(2) & np.isfinite(pred).all(2) & ~obs
    e = np.linalg.norm(pred - truth, axis=2)[ok]
    rad = np.array([[r if r is not None else np.nan for r in f.get("rad", [None] * truth.shape[1])] for f in alt], float)[ok]
    return {"hidden_n": int(ok.sum()), "mean_m": round(float(e.mean()), 3), "median_m": round(float(np.median(e)), 3),
            "within_5m": round(float((e <= 5).mean()), 3),
            "r80_coverage": round(float((e <= rad)[np.isfinite(rad)].mean()), 3) if np.isfinite(rad).any() else None}


def main(model, tag, file_tag=None):
    dfl_match.PIN_MODE = "off"
    weights = ROOT / "cv" / "weights" / "recon" / model
    log = []
    for i, mid in enumerate(dfl_matches()):
        for half in (1, 2):
            out = dfl_match.build(mid, half=half, seed=0, fold=f"fold{i}", weights_dir=weights)
            src = out["meta"]["source"]
            src.update(reconstruction_model=model, reconstruction_session=tag, reconstruction_fold=f"fold{i}",
                       event_pins="disabled (whole-half lag fit would leak future information into live)",
                       held_out_checked=True)
            path = DATA / f"dfl_{mid.split('-')[-1]}_{file_tag or tag}_h{half}.json"
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(out, separators=(",", ":")))
            tmp.replace(path)
            rec = {"match": mid, "half": half, "file": str(path.relative_to(ROOT)),
                   **{mode: hidden_metrics(out, mode) for mode in ("live", "replay")}}
            orig = DATA / f"dfl_{mid.split('-')[-1]}_h{half}.json"
            if orig.exists():
                o = json.loads(orig.read_text())
                rec["original_export"] = {mode: hidden_metrics(o, mode) for mode in ("live", "replay")}
            log.append(rec)
            print(json.dumps(rec), flush=True)
    (DATA / "eval" / f"apply_dfl_{tag}.json").write_text(json.dumps(log, indent=1))
    return log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--file-tag", default=None)
    a = ap.parse_args()
    main(a.model, a.tag, a.file_tag)
