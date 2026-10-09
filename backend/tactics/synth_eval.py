"""Quick synthetic tactical benchmark (full ground truth) over N simulator seeds, pooled precision / recall / F1.

    .venv/bin/python -m backend.tactics.synth_eval --n 20
"""
import argparse
from concurrent.futures import ProcessPoolExecutor


def _one(seed):
    from backend.models.match import MatchData
    from backend.reconstruction.evaluate import tactical_scores
    from simulator.match_generator import generate_match
    raw = generate_match(seed)
    return tactical_scores(MatchData(raw), raw["truth"])


def run(n=20, start=100, workers=6):
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_one, range(start, start + n)))
    out = {}
    for k in rows[0]:
        tp, fp, fn = (sum(r[k][x] for r in rows) for x in ("tp", "fp", "fn"))
        p, r = tp / max(1, tp + fp), tp / max(1, tp + fn)
        out[k] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / max(1e-9, p + r), 3), "tp": tp, "fp": fp, "fn": fn}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    for k, v in run(a.n, workers=a.workers).items():
        print(k, v)
