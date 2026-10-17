"""Re-infer hidden players in existing broadcast-derived app exports with a chosen model (additive, versioned).

Inputs are ONLY the stored measured positions (`obs` true); every stored inferred coordinate is discarded before
inference. Measured positions are copied unchanged. The output is written beside the original as
data/cv_<tag>_<name>.json with provenance; the original file is never modified.

  cv_2016_08_14_..._arsenal_...json  our own CV from SoccerNet broadcast video (role-based identity, ~35 % exact in an
                                     earlier evaluation): uncertainty radius scaled by inference.VIDEO_RADIUS_SCALE
  cv_skillcorner_epl_*_h*.json       SkillCorner legacy broadcast detections (provider identities)

    .venv/bin/python -m backend.reconstruction.apply_exports --model r13_final --tag r13
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from backend.reconstruction import hidden_models as hm
from backend.reconstruction.gsr_benchmark import model_path
from backend.reconstruction.inference import VIDEO_RADIUS_SCALE
from backend.reconstruction.skillcorner_benchmark import LINE_LAT, ROLE
from backend.reconstruction.skillcorner_match import reconstruct

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SOURCES = ["cv_2016_08_14_18_00_arsenal_3_4_liverpool_h1_1440_2040.json", "cv_skillcorner_epl_2440_h1.json",
           "cv_skillcorner_epl_2440_h2.json", "cv_skillcorner_epl_4039_h1.json", "cv_skillcorner_epl_4039_h2.json"]


def clip_from_export(d):
    F, order = d["frames"], d["meta"]["player_order"]
    T, N = len(F), len(order)
    pos = np.full((T, N, 2), np.nan)
    on = np.zeros((T, N), bool)
    for t, f in enumerate(F):
        for j, (p, o) in enumerate(zip(f["xy"], f["obs"])):
            if o is not None:
                on[t, j] = True
            if o is True and p is not None:
                pos[t, j] = p
    ball = np.array([f["b"] if f["b"] is not None else [np.nan, np.nan] for f in F], dtype=float)
    last = np.array([52.5, 34.0])
    for t in range(T):
        if np.isfinite(ball[t]).all():
            last = ball[t].copy()
        else:
            ball[t] = last
    players = {p["id"]: p for p in d["players"]}
    slots = [players[o].get("role_slot") or "" for o in order]
    is_gk = [players[o].get("position") == "GK" or s == "GK" for o, s in zip(order, slots)]
    acr = all(s in LINE_LAT or s == "GK" for s in slots)
    role = np.array(["GK" if g else (ROLE.get(s, "CM") if acr else "CM") for g, s in zip(is_gk, slots)])
    if acr:
        anchors = [((LINE_LAT.get(s, (2, 0))[0] - 1) / 2 * 30 - 15, -LINE_LAT.get(s, (2, 0))[1] * 22) for s in slots]
    else:
        from cv.clip_eval import coarse_roles
        cr = coarse_roles(d["players"])
        role = np.array([cr.get(o, "CM") for o in order])
        anchors = None
    return {"pos": pos, "on": on, "ball": ball, "team": np.array([o[0] for o in order]), "role": role,
            "poss": np.array([{"A": 0, "B": 1}.get(f.get("p"), -1) for f in F]), "alive": np.ones(T, bool),
            "anchors": anchors, "role_basis": "provider role acronyms" if acr else "lineup slot (cv.clip_eval.coarse_roles)"}


def apply(src, models, model_name, tag, file_tag=None, out_path=None):
    d = json.loads((DATA / src).read_text())
    sha = hashlib.sha256((DATA / src).read_bytes()).hexdigest()
    clip = clip_from_export(d)
    recs = reconstruct(clip, models)
    video = "arsenal" in src
    scale = VIDEO_RADIUS_SCALE if video else 1.0

    def frames(rec):
        pos, r80, p5, vis = rec
        out = []
        for t, f in enumerate(d["frames"]):
            g = dict(f)
            xy, rad, conf = list(f["xy"]), [], []
            for j, o in enumerate(f["obs"]):
                if o is None:
                    rad.append(None), conf.append(None)
                elif o is True and f["xy"][j] is not None:
                    rad.append(0.0), conf.append(1.0)  # measured coordinate copied unchanged
                else:
                    xy[j] = [round(float(pos[t, j, 0]), 2), round(float(pos[t, j, 1]), 2)]
                    rad.append(round(float(r80[t, j]) * scale, 1)), conf.append(round(float(p5[t, j]), 2))
            g.update(xy=xy, rad=rad, conf=conf)
            out.append(g)
        return out
    replay, live = frames(recs["replay"]), frames(recs["live"])
    out = dict(d)
    out["frames"] = replay
    out["alt_frames"] = {"replay": replay, "live": live}
    src_meta = dict(d["meta"].get("source") or {})
    src_meta.update(reconstruction_model=model_name, reconstruction_session=tag, derived_from=src, derived_from_sha256=sha,
                    reconstruction_inputs="stored measured positions only; stored inferred coordinates discarded",
                    role_basis=clip["role_basis"], live_available=True,
                    radius_scale=scale, uncertainty_basis=("DFL-calibrated radius x %.2f (video-domain scale fitted on this clip's first half in an earlier session)" % scale
                                                           if video else "DFL-calibrated radius; coverage on SkillCorner A-League re-entries reported in docs/reconstruction_evaluation.md"))
    src_meta.pop("error_vs_truth", None)
    out["meta"] = dict(d["meta"], source=src_meta)
    name = src.removeprefix("cv_")
    path = Path(out_path) if out_path else DATA / f"cv_{file_tag or tag}_{name}"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, separators=(",", ":")))
    tmp.replace(path)
    # integrity: every measured coordinate unchanged
    for t, f in enumerate(d["frames"]):
        for j, o in enumerate(f["obs"]):
            if o is True and f["xy"][j] is not None:
                assert out["frames"][t]["xy"][j] == f["xy"][j]
    hid = int(sum((o is False) for f in d["frames"] for o in f["obs"]))
    return {"source": src, "output": str(path.relative_to(ROOT)), "hidden_player_seconds": hid,
            "observed_player_seconds": int(sum((o is True) for f in d["frames"] for o in f["obs"])), "source_sha256": sha}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--file-tag", default=None, help="keep an existing export's file name (app id), e.g. r13")
    a = ap.parse_args()
    models = {mode: hm.load_hidden(model_path(a.model, mode)) for mode in ("live", "replay")}
    log = [apply(s, models, a.model, a.tag, a.file_tag) for s in SOURCES]
    print(json.dumps(log, indent=1))
    (DATA / "eval" / f"apply_exports_{a.tag}.json").write_text(json.dumps(log, indent=1))
