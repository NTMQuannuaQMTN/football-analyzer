"""SoccerNet Game State Reconstruction (GSR): load clips, evaluate our pipeline, export training data.

Each clip: 750 frames (1080p, 25 fps) with every person's box, pitch position (metres, centre
origin, +y towards the camera), role, team (left/right = goal defended) and jersey number.

    .venv/bin/python -m cv.gsr eval --split valid --clips 12 --stride 10 --model cv/weights/yolo11m.pt
"""
import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from cv.build_match import kmeans
from cv.detect_track import shirt_color

ROOT = Path(__file__).resolve().parents[1]
GSR = ROOT / "data" / "sn_gsr"
OUT = Path(__file__).resolve().parent / "out" / "gsr"
L, W = 105.0, 68.0
MATCH_M = 5.0  # GS-HOTA localisation tolerance


def clips(split):
    return sorted(p for p in (GSR / split).iterdir() if (p / "Labels-GameState.json").exists())


def load_clip(clip: Path):
    d = json.load(open(clip / "Labels-GameState.json"))
    frames = {}
    for im in d["images"]:
        frames[im["image_id"]] = {"file": clip / "img1" / im["file_name"], "w": im["width"], "h": im["height"], "people": []}
    for a in d["annotations"]:
        if a["supercategory"] != "object" or a["image_id"] not in frames:
            continue
        b = a["bbox_image"]
        p = a.get("bbox_pitch") or {}
        px, py = p.get("x_bottom_middle"), p.get("y_bottom_middle")
        ok = px is not None and abs(px) < 60 and abs(py) < 40
        frames[a["image_id"]]["people"].append({
            "track": a["track_id"], "role": a["attributes"]["role"], "team": a["attributes"]["team"],
            "jersey": a["attributes"]["jersey"], "box": (b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]),
            "pitch": (px + L / 2, py + W / 2) if ok else None})
    return d["info"], [frames[k] for k in sorted(frames)]


def oracle_homography(people):
    """Image->pitch homography fitted to the labelled feet of this frame (isolates calibration error)."""
    src, dst = [], []
    for p in people:
        if p["pitch"] is not None and p["role"] != "ball":
            x1, y1, x2, y2 = p["box"]
            src.append(((x1 + x2) / 2, y2))
            dst.append(p["pitch"])
    if len(src) < 5:
        return None
    H, _ = cv2.findHomography(np.float32(src), np.float32(dst), cv2.RANSAC, 1.5)
    return H


def project(H, pts):
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    hp = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return hp[:, :2] / hp[:, 2:3]


def evaluate(split, n_clips, stride, model_path, device):
    from ultralytics import YOLO
    model = YOLO(model_path)
    names = model.names
    person_cls = [k for k, v in names.items() if v in ("person", "player", "goalkeeper", "referee")]
    ball_cls = [k for k, v in names.items() if v in ("sports ball", "ball")]
    stats = Counter()
    loc_err, team_rows = [], []
    t0 = time.time()
    for clip in clips(split)[:n_clips]:
        info, frames = load_clip(clip)
        clip_team = []  # (colour feature, gt team) for matched outfield players
        for f in frames[::stride]:
            H = oracle_homography(f["people"])
            if H is None:
                continue
            img = cv2.imread(str(f["file"]))
            res = model.predict(img, device=device, verbose=False, imgsz=1280, conf=0.03, classes=person_cls + ball_cls)[0]
            boxes = res.boxes.xyxy.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            conf = res.boxes.conf.cpu().numpy()
            # --- people
            pm = [(b, c) for b, c, s in zip(boxes, cls, conf) if c in person_cls and s >= 0.25]
            gt = [p for p in f["people"] if p["role"] != "ball" and p["pitch"] is not None]
            if pm and gt:
                pp = project(H, [((b[0] + b[2]) / 2, b[3]) for b, _ in pm])
                G = np.array([p["pitch"] for p in gt])
                C = np.linalg.norm(pp[:, None] - G[None], axis=2)
                r, c = linear_sum_assignment(C)
                hit = [(a, g) for a, g in zip(r, c) if C[a, g] <= MATCH_M]
                stats["tp"] += len(hit)
                stats["fp"] += len(pm) - len(hit)
                stats["fn"] += len(gt) - len(hit)
                loc_err += [float(C[a, g]) for a, g in hit]
                for a, g in hit:
                    if gt[g]["role"] == "player" and gt[g]["team"] in ("left", "right"):
                        col = shirt_color(img, pm[a][0])
                        if col:
                            c1, c2 = col[0], col[1] if len(col) > 1 else col[0]
                            clip_team.append(([c1[0] * .35, c1[1], c1[2], c2[0] * .35, c2[1], c2[2]], gt[g]["team"]))
            else:
                stats["fn"] += len(gt)
                stats["fp"] += len(pm)
            # --- ball (image space: centre within 20 px at 1080p)
            gb = [p for p in f["people"] if p["role"] == "ball"]
            if gb:
                stats["ball_gt"] += 1
                bx = [b for b, c, s in zip(boxes, cls, conf) if c in ball_cls]
                if bx:
                    gx = ((gb[0]["box"][0] + gb[0]["box"][2]) / 2, (gb[0]["box"][1] + gb[0]["box"][3]) / 2)
                    best = min(np.hypot((b[0] + b[2]) / 2 - gx[0], (b[1] + b[3]) / 2 - gx[1]) for b in bx)
                    stats["ball_hit"] += int(best <= 20)
            stats["frames"] += 1
        # --- team assignment: unsupervised colour clusters vs ground truth (best permutation)
        if len(clip_team) >= 20:
            X = np.array([c for c, _ in clip_team])
            lab, _ = kmeans(X, 3)
            gtt = [t for _, t in clip_team]
            big = [j for j, _ in Counter(lab).most_common(2)]
            best = 0
            for perm in ((big[0], big[1]), (big[1], big[0])):
                m = {perm[0]: "left", perm[1]: "right"}
                best = max(best, sum(m.get(l) == t for l, t in zip(lab, gtt)))
            team_rows.append(best / len(gtt))
        print(f"{clip.name}: frames {stats['frames']} ({time.time() - t0:.0f}s)", flush=True)
    prec = stats["tp"] / max(1, stats["tp"] + stats["fp"])
    rec = stats["tp"] / max(1, stats["tp"] + stats["fn"])
    out = {"model": str(model_path), "split": split, "clips": n_clips, "frames": stats["frames"],
           "people": {"precision": round(prec, 3), "recall": round(rec, 3), "mean_loc_error_m": round(float(np.mean(loc_err)), 2)},
           "ball_recall_20px": round(stats["ball_hit"] / max(1, stats["ball_gt"]), 3),
           "team_accuracy": round(float(np.mean(team_rows)), 3) if team_rows else None,
           "jersey_accuracy": 0.0,
           "note": "oracle calibration (homography fitted to labelled feet) isolates detection/team quality"}
    OUT.mkdir(parents=True, exist_ok=True)
    tag = Path(model_path).stem
    json.dump(out, open(OUT / f"eval_{split}_{tag}.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["eval"])
    ap.add_argument("--split", default="valid")
    ap.add_argument("--clips", type=int, default=12)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--model", default=str(Path(__file__).resolve().parent / "weights" / "yolo11m.pt"))
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    evaluate(a.split, a.clips, a.stride, a.model, a.device)
