"""Prediction -> reality over every DFL half file: the live model's prediction at the second a hidden player came
back into the camera view, vs where he really was. Only matches the estimator in the file never trained on count
(the app files use fold-0/1 models: Köln-Bayern and Bochum-Leverkusen are their held-out matches).

    .venv/bin/python -m backend.reconstruction.reveal_report      # data/eval/real/reveals_report.json
"""
import glob
import json

import numpy as np

from backend.reconstruction.datasets import CACHE, ROOT


def main():
    rows = []
    for f in sorted(glob.glob(str(ROOT / "data" / "dfl_*_h[12].json"))):
        d = json.load(open(f))
        if d["meta"]["source"].get("estimator_saw_this_match"):
            continue
        rows += d.get("reveals", [])
    e = np.array([r["error_m"] for r in rows])
    hs = np.array([r["hidden_s"] for r in rows])
    out = {"matches": "held-out only (estimator never trained on them)", "n": len(rows),
           "mean_error_m": round(float(e.mean()), 2), "median_error_m": round(float(np.median(e)), 2),
           "within_5m": round(float((e <= 5).mean()), 3), "within_10m": round(float((e <= 10).mean()), 3),
           "inside_80pct_radius": round(float(np.mean([r["inside_radius"] for r in rows])), 3),
           "by_hidden_s": {f"{a}-{b if b < 9999 else ''}s": {"n": int(((hs >= a) & (hs < b)).sum()),
                                                            "median_error_m": round(float(np.median(e[(hs >= a) & (hs < b)])), 2),
                                                            "inside_80pct_radius": round(float(np.mean([r["inside_radius"] for r, h in zip(rows, hs) if a <= h < b])), 3)}
                           for a, b in ((5, 10), (10, 30), (30, 9999))}}
    json.dump(out, open(CACHE / "reveals_report.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
