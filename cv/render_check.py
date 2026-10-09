"""Visual check: broadcast frame with our ball/players drawn on it, next to our minimap.

    .venv/bin/python -m cv.render_check --start 1800 --end 1815      # writes cv/out/check_<start>.mp4 + PNGs

Each detected player's foot point and the ball are re-projected from pitch metres back into the
image with the same calibration, so a mismatch between video and minimap is visible immediately.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from cv.detect_track import OUT, slug
from cv.soccernet import DEFAULT_GAME, Calibration, game_dir

ROOT = Path(__file__).resolve().parents[1]


def minimap(frame_rec, meta, order, scale=6):
    L, W = 105, 68
    img = np.zeros((W * scale + 20, L * scale + 20, 3), np.uint8)
    img[:] = (40, 110, 40)
    cv2.rectangle(img, (10, 10), (10 + L * scale, 10 + W * scale), (255, 255, 255), 1)
    cv2.line(img, (10 + L * scale // 2, 10), (10 + L * scale // 2, 10 + W * scale), (255, 255, 255), 1)
    hexc = lambda h: tuple(int(h[i:i + 2], 16) for i in (5, 3, 1))
    for pid, p, o in zip(order, frame_rec["xy"], frame_rec.get("obs") or [True] * len(order)):
        if p is None:
            continue
        c = hexc(meta["teams"][pid[0]]["color"])
        pt = (int(10 + p[0] * scale), int(10 + p[1] * scale))
        cv2.circle(img, pt, 8, c, -1 if o else 2)
        cv2.putText(img, pid[1:], (pt[0] - 7, pt[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    b = frame_rec["b"]
    cv2.circle(img, (int(10 + b[0] * scale), int(10 + b[1] * scale)), 5, (0, 255, 255), -1)
    if frame_rec.get("c"):
        cv2.putText(img, f"owner {frame_rec['c']}", (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return img


def run(game, half, start, end, run_start, run_end):
    match = json.load(open(ROOT / "data" / f"cv_{slug(game, half, run_start, run_end)}.json"))
    meta, order = match["meta"], match["meta"]["player_order"]
    dets = {round(json.loads(l)["t"], 1): json.loads(l) for l in open(OUT / slug(game, half, run_start, run_end) / "detections.jsonl")}
    calib = Calibration(game, half)
    cap = cv2.VideoCapture(str(game_dir(game) / f"{half}_720p.mkv"))
    out_dir = OUT / "check"
    out_dir.mkdir(parents=True, exist_ok=True)
    vw = None
    for t in np.arange(start, end, 0.2):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok:
            break
        fr = cv2.resize(fr, (1280, 720))
        H = calib.at(t)
        d = dets.get(round(t, 1))
        if d:
            for x in d["dets"]:
                x1, y1, x2, y2 = [int(v) for v in x["box"]]
                col = (0, 255, 255) if x["cls"] == "ball" else (255, 255, 255)
                cv2.rectangle(fr, (x1, y1), (x2, y2), col, 1)
        s = int(t - run_start)
        rec = match["frames"][min(s, len(match["frames"]) - 1)]
        if H is not None:
            Hinv = np.linalg.inv(H)  # pitch -> image
            def to_img(p):
                v = Hinv @ np.array([p[0], p[1], 1.0])
                return int(v[0] / v[2]), int(v[1] / v[2])
            for pid, p, o in zip(order, rec["xy"], rec.get("obs") or [True] * len(order)):
                if p is not None and o:
                    cv2.circle(fr, to_img(p), 5, (0, 0, 255), 2)
            bx = to_img(rec["b"])
            cv2.drawMarker(fr, bx, (0, 255, 0), cv2.MARKER_CROSS, 22, 2)
        cv2.putText(fr, f"{int(t // 60)}:{t % 60:04.1f}  live={d['live'] if d else '?'}  yellow=detector box  green+=our ball  red=our players",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        mm = minimap(rec, meta, order)
        mm = cv2.resize(mm, (int(mm.shape[1] * 720 / mm.shape[0]), 720))
        combo = np.hstack([fr, mm])
        if vw is None:
            vw = cv2.VideoWriter(str(out_dir / f"check_{int(start)}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 5, (combo.shape[1], combo.shape[0]))
        vw.write(combo)
        if abs((t - start) % 2.0) < 0.1:
            cv2.imwrite(str(out_dir / f"check_{t:.1f}.jpg"), cv2.resize(combo, None, fx=0.6, fy=0.6))
    if vw:
        vw.release()
    print("wrote", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default=DEFAULT_GAME)
    ap.add_argument("--half", type=int, default=1)
    ap.add_argument("--start", type=float, default=1800)
    ap.add_argument("--end", type=float, default=1815)
    ap.add_argument("--run-start", type=float, default=1440)
    ap.add_argument("--run-end", type=float, default=2040)
    a = ap.parse_args()
    run(a.game, a.half, a.start, a.end, a.run_start, a.run_end)
