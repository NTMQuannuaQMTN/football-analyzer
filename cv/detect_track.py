"""Steps 1-5: detect + track players/ball on broadcast video and project them to pitch metres.

Output: cv/out/<slug>/detections.jsonl, one line per sampled frame:
  {"t": seconds_in_half, "live": bool, "dets": [{"id", "cls", "conf", "box", "px", "py", "color"}]}
"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np

from cv.soccernet import DEFAULT_GAME, Calibration, camera_segments, game_dir, labels, on_main_camera
from cv.track_namespace import TrackNamespace

OUT = Path(__file__).resolve().parent / "out"
TRACKER = Path(__file__).resolve().parent / "botsort_broadcast.yaml"
PERSON_NAMES = ("person", "player", "goalkeeper", "referee")  # COCO "person" or fine-tuned SoccerNet roles
BALL_NAMES = ("sports ball", "ball")


def slug(game: str, half: int, start: float, end: float) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", game.split("/")[-1].lower()).strip("_")
    return f"{s}_h{half}_{int(start)}_{int(end)}"


def replay_segments(game, half):
    return [(start,end) for start,end,a in camera_segments(game,half) if a.get("replay")=="replay"]


def green_ratio(frame) -> float:
    hsv = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2HSV)
    m = (hsv[..., 0] > 30) & (hsv[..., 0] < 90) & (hsv[..., 1] > 50) & (hsv[..., 2] > 40)
    return float(m.mean())


def shirt_color(frame, box):
    """Two dominant non-grass torso colours (Lab + weight), most chromatic first.

    A single median blends shirt body, sleeves and shadow into mud; two clusters keep e.g.
    Arsenal's red body separate from the white sleeves.
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = y2 - y1, x2 - x1
    crop = frame[max(0, y1 + int(0.18 * h)):max(0, y1 + int(0.48 * h)), max(0, x1 + int(0.2 * w)):max(0, x2 - int(0.2 * w))]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    grass = (hsv[..., 0] > 30) & (hsv[..., 0] < 90) & (hsv[..., 1] > 50)
    px = crop[~grass]
    if len(px) < 8:
        return None
    lab = cv2.cvtColor(px.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    k = 2 if len(lab) >= 16 else 1
    _, lbl, centers = cv2.kmeans(lab, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 2, cv2.KMEANS_PP_CENTERS)
    out = []
    for j in range(k):
        wgt = float(np.mean(lbl.ravel() == j))
        out.append([round(float(v), 1) for v in centers[j]] + [round(wgt, 2)])
    out.sort(key=lambda c: -np.hypot(c[1] - 128, c[2] - 128))  # most chromatic first
    return out


def is_live_view(frame, H, boxes) -> bool:
    if H is None or green_ratio(frame) < 0.35:
        return False
    c = Calibration.project(H, [[640, 360]])[0]
    if not (-10 < c[0] < 115 and -10 < c[1] < 78):
        return False
    heights = [b[3] - b[1] for b in boxes]
    return not heights or float(np.median(heights)) < 160


def run(game, half, start, end, fps, model_name, device, show, output_dir=None, seed=0,tracker_path=None,calibration_offset=0.):
    from ultralytics import YOLO
    tracker=Path(tracker_path) if tracker_path is not None else TRACKER
    if not tracker.is_file():raise FileNotFoundError(f'Missing tracker configuration: {tracker}')
    cv2.setRNGSeed(seed)
    video = game_dir(game) / f"{half}_720p.mkv"
    if not video.exists():
        raise SystemExit(f"Missing {video}\nRun: hf auth login && .venv/bin/python -m cv.soccernet --game '{game}' --half {half}")
    calib = Calibration(game, half)
    replays = replay_segments(game, half)
    model = YOLO(model_name)
    person_ids = [k for k, v in model.names.items() if v in PERSON_NAMES]
    ball_ids = [k for k, v in model.names.items() if v in BALL_NAMES]
    # Isolate ball prediction arguments and model state. The installed
    # Ultralytics guards tracker callbacks by mode; an audit did NOT reproduce
    # a double tracker update, so this has no measured accuracy benefit.
    ball_model = YOLO(model_name) if ball_ids else None
    cap = cv2.VideoCapture(str(video))
    vfps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(vfps / fps)))
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    out_dir = Path(output_dir) if output_dir else OUT / slug(game, half, start, end)
    out_dir.mkdir(parents=True, exist_ok=True)
    f = open(out_dir / "detections.jsonl", "w")
    n, t0, was_live = 0, time.time(), False
    track_namespace=TrackNamespace()
    idx = int(round(start * vfps))
    while True:
        ok = cap.grab()
        if not ok:
            break
        t = idx / vfps
        idx += 1
        if t > end:
            break
        if (idx - 1) % step:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        if frame.shape[1] != 1280:
            frame = cv2.resize(frame, (1280, 720))
        H = calib.at(t+calibration_offset)
        in_replay = any(a <= t < b for a, b in replays)
        if in_replay or not on_main_camera(game,half,t):
            # Replay/close-up detections belong to another time or camera.
            # They must not update the player tracker or enter observations.
            if getattr(model,"predictor",None) is not None and hasattr(model.predictor,"trackers"):
                model.predictor.trackers[0].reset()
            was_live=False
            track_namespace.camera(False)
            f.write(json.dumps({"t":round(t,2),"live":False,"dets":[],"camera_basis":"shot-end annotations"})+"\n")
            n+=1
            continue
        # players: tracked (BoT-SORT with camera-motion compensation)
        res = model.track(frame, persist=True, classes=person_ids, tracker=str(tracker), device=device,
                          verbose=False, imgsz=1280, conf=0.15, agnostic_nms=True)[0]
        # ball: plain detection at low confidence — the tracker discards small fast low-score objects
        bres = ball_model.predict(frame, classes=ball_ids, device=device, verbose=False, imgsz=1280, conf=0.03)[0] if ball_model else None
        boxes, cls, confs, ids = [], [], [], []
        if res.boxes is not None and len(res.boxes):
            boxes += list(res.boxes.xyxy.cpu().numpy())
            cls += list(res.boxes.cls.cpu().numpy().astype(int))
            confs += list(res.boxes.conf.cpu().numpy())
            ids += list(res.boxes.id.cpu().numpy().astype(int)) if res.boxes.id is not None else [-1] * len(res.boxes)
        if bres is not None and bres.boxes is not None and len(bres.boxes):
            # keep the top-3 ball candidates; build_match picks the physically consistent one
            for j in bres.boxes.conf.argsort(descending=True)[:3].tolist():
                boxes.append(bres.boxes.xyxy[j].cpu().numpy())
                cls.append(ball_ids[0])
                confs.append(float(bres.boxes.conf[j]))
                ids.append(-1)
        boxes = np.array(boxes).reshape(-1, 4)
        live = (not in_replay) and is_live_view(frame, H, [b for b, c in zip(boxes, cls) if c in person_ids])
        if not live and was_live:
            model.predictor.trackers[0].reset()  # camera cut: don't carry IDs across shots
        was_live = live
        track_namespace.camera(live)
        dets = []
        if live:
            feet = np.array([[(b[0] + b[2]) / 2, b[3] if c in person_ids else (b[1] + b[3]) / 2] for b, c in zip(boxes, cls)]) if len(boxes) else np.zeros((0, 2))
            pitch = Calibration.project(H, feet) if len(feet) else np.zeros((0, 2))
            for b, c, cf, i, p in zip(boxes, cls, confs, ids, pitch):
                if not (-3 < p[0] < 108 and -3 < p[1] < 71):
                    continue  # crowd, bench, staff
                role = model.names[int(c)]
                kind = "ball" if c in ball_ids else ("referee" if role == "referee" else "person")
                d = {"id": int(i), "cls": kind, "role": "player" if role == "person" else role, "conf": round(float(cf), 3),
                     "box": [round(float(v), 1) for v in b], "px": round(float(p[0]), 2), "py": round(float(p[1]), 2)}
                if kind in ('person','referee') and i>=0:
                    d.update(id=track_namespace.identity(i),native_id=int(i),shot_id=track_namespace.shot)
                if kind == "person":
                    d["colors"] = shirt_color(frame, b)
                dets.append(d)
        f.write(json.dumps({"t": round(t, 2), "live": live, "dets": dets}) + "\n")
        n += 1
        if show:
            vis = bres.plot(img=res.plot()) if bres is not None else res.plot()
            cv2.putText(vis, f"{t:.1f}s live={live}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("track", vis)
            if cv2.waitKey(1) == 27:
                break
        if n % 100 == 0:
            print(f"t={t:.0f}s frames={n} {n / (time.time() - t0):.1f} fps")
    f.close()
    meta = {"game": game, "half": half, "start": start, "end": end, "fps": fps, "model": model_name,"colour_seed":seed,
            "tracker_path":str(tracker),"tracker_sha256":hashlib.sha256(tracker.read_bytes()).hexdigest(),
            "track_id_namespace":"shot-qualified-v1","person_nms":"class-agnostic; player/keeper/referee labels do not create separate people",
            "calibration_offset_s":calibration_offset,"calibration_sampling":"retrospective nearest 2Hz CCBV",
            "camera_labels":"shot-end intervals; replays excluded before tracking", "retrospective":True}
    json.dump(meta, open(out_dir / "run.json", "w"), indent=1)
    print("wrote", out_dir)
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default=DEFAULT_GAME)
    ap.add_argument("--half", type=int, default=1)
    ap.add_argument("--start", type=float, default=24 * 60, help="seconds into the half")
    ap.add_argument("--end", type=float, default=34 * 60)
    ap.add_argument("--fps", type=float, default=5)
    ap.add_argument("--model", default=str(Path(__file__).resolve().parent / "weights" / "yolo_gsr.pt"),
                    help="fine-tuned on SoccerNet-GSR (player/goalkeeper/referee/ball); yolo11m.pt = stock COCO")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--show", action="store_true", help="live preview window")
    ap.add_argument("--out", help="isolated output directory (preserves existing detections)")
    ap.add_argument("--tracker",help="Explicit tracker YAML; defaults to the preserved production configuration")
    ap.add_argument("--seed",type=int,default=0,help="Fixed colour-clustering seed")
    ap.add_argument('--calibration-offset',type=float,default=0.,help='Experimental camera/video offset; requires independent validation')
    a = ap.parse_args()
    _, labs = labels(a.game, a.half)
    print("events in window:", [(round(x["t"]), x["label"], x["team"]) for x in labs if a.start <= x["t"] <= a.end and x["label"] in ("Goal", "Shots on target", "Shots off target", "Penalty")])
    run(a.game, a.half, a.start, a.end, a.fps, a.model, a.device, a.show, a.out,a.seed,a.tracker,a.calibration_offset)
