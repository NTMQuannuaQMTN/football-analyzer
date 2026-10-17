"""Precomputed analysis of the synthetic match, shipped with the deployment.

Building the synthetic match costs ~60 s of CPU (the simulator runs once for the match and once per counterfactual,
then reconstruction and analytics). It is deterministic for a seed, so it is built once and loaded on start (~0.3 s).
The file records a digest of the analysis code and of the settings that change it (VIEW); a file built by other code
is ignored and the match is rebuilt as before, so a stale cache can cost time but never serve a stale analysis.

Rebuild after changing backend/ or simulator/ (run.sh does it when the file is stale):
    .venv/bin/python -m backend.precomputed
"""
import gzip
import hashlib
import os
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data" / "precomputed"


def code_digest() -> str:
    h = hashlib.sha256()
    for base in ("backend", "simulator"):
        for p in sorted((ROOT / base).rglob("*.py")):
            h.update(str(p.relative_to(ROOT)).encode())
            h.update(p.read_bytes())
    h.update(f"VIEW={os.getenv('VIEW', 'truth')}".encode())
    return h.hexdigest()


def path(seed) -> Path:
    """seed: the synthetic match with this seed; a str: the analysis of that match file."""
    return DIR / (f"intel_seed{seed}.pkl.gz" if isinstance(seed, int) else f"intel_{Path(seed).stem}.pkl.gz")


def file_digest(src) -> str:
    """Code digest + the match file's content: a re-exported file never gets the old analysis."""
    return hashlib.sha256((code_digest() + hashlib.sha256(Path(src).read_bytes()).hexdigest()).encode()).hexdigest()


def load_file(src):
    """The precomputed MatchIntelligence for this match file, or None."""
    return load(str(src), file_digest(src)) if path(str(src)).exists() else None


def load(seed, digest: str = None):
    """The precomputed MatchIntelligence for this seed (or file), or None (missing, unreadable, or built by other code)."""
    p = path(seed)
    if not p.exists():
        return None
    try:
        with gzip.open(p, "rb") as f:
            rec = pickle.load(f)  # our own build artefact, written by save() below
    except Exception:
        return None
    return rec["intel"] if rec.get("digest") == (digest or code_digest()) else None


def save(seed, intel, digest: str = None) -> Path:
    DIR.mkdir(parents=True, exist_ok=True)
    p = path(seed)
    tmp = p.with_suffix(".tmp")
    with gzip.open(tmp, "wb", compresslevel=6) as f:
        pickle.dump({"digest": digest or code_digest(), "seed": seed, "intel": intel}, f, protocol=5)
    tmp.replace(p)
    return p


def warm(I):
    """Fill the lazy caches the first requests would otherwise pay for (per-second formation lines ~10 s)."""
    for mode in ("default", "replay", "live", "truth"):
        try:
            I.frames(mode)
        except Exception:
            pass  # a mode this match does not have
    I.summary()


def fresh(seed: int) -> bool:
    return load(seed) is not None


# match files shipped with the deployment (git-tracked, CC BY 4.0): their analysis is precomputed too
SHIPPED_FILES = ["data/dfl_J03WMX_r13_h1.json", "data/dfl_J03WMX_r13_h2.json"]


if __name__ == "__main__":
    import argparse
    import time

    from backend.pipeline import MatchIntelligence
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--if-stale", action="store_true", help="only rebuild when the shipped file does not match the code")
    a = ap.parse_args()
    jobs = [(a.seed, None)] + [(str((ROOT / f).resolve()), f) for f in SHIPPED_FILES if (ROOT / f).exists()]
    for key, src in jobs:
        if a.if_stale and (fresh(key) if src is None else load_file(key) is not None):
            print(f"{path(key).relative_to(ROOT)} is up to date")
            continue
        t0 = time.time()
        I = MatchIntelligence(a.seed, key) if src else MatchIntelligence(a.seed, None)
        warm(I)
        p = save(key, I, file_digest(key) if src else None)
        print(f"wrote {p.relative_to(ROOT)} ({p.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s")
