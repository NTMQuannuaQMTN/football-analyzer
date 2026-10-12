"""Extras for a computer-vision match file (cv/build_match.py output), so the studio can show "what the camera
couldn't see" on real broadcast video:

  frames[t]["cam"]  the broadcast frame's footprint on the pitch, from the SoccerNet calibration (image borders
                    projected to metres; parts above the horizon are cut at the far touchline)
  reveals           prediction -> reality: for each player who reappears after >= 5 s off screen, the live (causal)
                    prediction made with that second masked vs where the video then detects him. "Actual" is a CV
                    detection and identities are role-based, so these errors are an upper bound.

    .venv/bin/python -m cv.clip_extras data/cv_<clip>.json
"""
import json
import sys

import numpy as np

from backend.models.match import L, W

IMG_W, IMG_H = 1280, 720


def footprint(H, nx=32, ny=18):
    try:
        inverse = np.linalg.inv(H)
        ref = np.asarray(H) @ [IMG_W/2, IMG_H-1, 1]
        if abs(ref[2]) < 1e-10:
            return None
        ref = ref/ref[2]
        if (inverse @ ref)[2] < 0:
            inverse = -inverse
    except np.linalg.LinAlgError:
        return None
    # Exact intersection of pitch and image frustum. Grid hulls underestimated
    # edge coverage and depended on the arbitrary sign of homogeneous H.
    polygon = np.array([[0.,0.],[L,0.],[L,W],[0.,W]])
    r0,r1,r2 = inverse
    for plane in (r2,r0,IMG_W*r2-r0,r1,IMG_H*r2-r1):
        if len(polygon)<3:
            return None
        result=[]
        for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
            da,db = plane @ [*a,1],plane @ [*b,1]
            if da>=0:
                result.append(a)
            if (da>=0)!=(db>=0):
                result.append(a+(b-a)*da/(da-db))
        polygon=np.asarray(result)
    return polygon.tolist() if len(polygon)>=3 else None


def cached_camera_matrices(run_dir):
    from pathlib import Path
    root=Path(run_dir);meta=json.loads((root/'run.json').read_text())
    if meta.get('camera_H_coordinates')!='normalised_0_to_1_image_to_pitch_metres':
        raise ValueError('Candidate camera units must be explicit')
    result={}
    for line in (root/'detections.jsonl').open():
        row=json.loads(line);h=row.get('H');t=round(float(row['t']),6)
        if t in result:raise ValueError('Duplicate candidate camera timestamp')
        result[t]=None if h is None else np.asarray(h,float)@np.diag([1/IMG_W,1/IMG_H,1.])
    return result


def add_extras(path):
    from backend.reconstruction.inference import VIDEO_RADIUS_SCALE, reconstruct_ml
    from cv.clip_eval import coarse_roles, load, reveals
    from cv.soccernet import Calibration, on_main_camera
    d = json.load(open(path))
    src = d["meta"]["source"]
    calib = Calibration(src["game"], src["half"])
    candidate=cached_camera_matrices(src['camera_run']) if src.get('camera_run') else None
    n_cam = 0
    for t, f in enumerate(d["frames"]):
        H = candidate.get(round(src['start']+t,6)) if candidate is not None else calib.at(src["start"] + t+src.get('calibration_offset_s',0.)) if on_main_camera(src["game"],src["half"],src["start"]+t) else None
        poly = footprint(H) if H is not None else None
        if poly:
            f["cam"] = poly
            f['camera_basis']='Candidate camera used for measurement projection; unverified' if candidate is not None else 'Estimated CCBV footprint; calibration and alignment unverified'
            n_cam += 1
        else:
            f.pop("cam", None)
            f.pop('camera_basis',None)
    # prediction -> reality on video
    _, m, vis, on = load(path)
    rv = reveals(vis, on)
    masked = vis.copy()
    for t, j, _ in rv:
        masked[t, j] = False
    teams = np.array([o[0] for o in m.order])
    cr = coarse_roles(d["players"])
    rc = np.array([cr[o] for o in m.order])
    role = {o: ("GK" if cr[o] == "GK" else "CM") for o in m.order}
    rec = reconstruct_ml(m, masked, "live", on, role, None, teams, rc, radius_scale=VIDEO_RADIUS_SCALE,
                         fold=src.get("reconstruction_fold"), weights_dir=src.get("reconstruction_weights"))
    out = []
    for t, j, h in rv:
        pred, act = rec.pos[t, j], m.xy[t, j]
        if not (np.isfinite(pred).all() and np.isfinite(act).all()):
            continue
        err = float(np.hypot(*(pred - act)))
        out.append({"t": int(t), "player_id": m.order[j], "hidden_s": int(h),
                    "predicted": [round(float(pred[0]), 1), round(float(pred[1]), 1)],
                    "actual": [round(float(act[0]), 1), round(float(act[1]), 1)], "error_m": round(err, 2),
                    "radius_m": round(float(rec.rad[t, j]), 1), "confidence": round(float(rec.conf[t, j]), 2),
                    "inside_radius": bool(err <= rec.rad[t, j])})
    d["reveals"] = out
    if out:
        e = np.array([r["error_m"] for r in out])
        src["error_vs_cv_proxy"] = {"reveals": {
            "n": len(out), "mean_error_m": round(float(e.mean()), 2), "median_error_m": round(float(np.median(e)), 2),
            "inside_80pct_radius": round(float(np.mean([r["inside_radius"] for r in out])), 3), "by_hidden_s": {},
            "note": "Reference is a role-assigned CV detection, NOT ground truth. This is a causal-feature reappearance consistency diagnostic on retrospectively processed inputs, not validated live video accuracy. Radius multiplier %.2f is transferred from an older model and is unvalidated for this candidate." % VIDEO_RADIUS_SCALE}}
    json.dump(d, open(path, "w"))
    print(f"clip extras: camera footprint in {n_cam}/{len(d['frames'])} s, {len(out)} reappearance tests")
    return d


if __name__ == "__main__":
    add_extras(sys.argv[1] if len(sys.argv) > 1 else "data/cv_2016_08_14_18_00_arsenal_3_4_liverpool_h1_1440_2040.json")
