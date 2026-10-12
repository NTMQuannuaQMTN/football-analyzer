"""Direct local-video check that ball inference does not advance player state."""
import json
from pathlib import Path

import cv2
import torch

from cv.detect_track import TRACKER
from cv.soccernet import DEFAULT_GAME,game_dir


def main():
    from ultralytics import YOLO
    torch.set_num_threads(1)
    cap=cv2.VideoCapture(str(game_dir(DEFAULT_GAME)/"1_720p.mkv"))
    frames=[]
    for t in [1800+i*.2 for i in range(4)]:
        cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,frame=cap.read()
        if not ok:
            raise RuntimeError(f"Cannot read video at {t}")
        frames.append(frame)
    cap.release(); report={}
    for shared in (True,False):
        person=YOLO("cv/weights/yolo11m.pt")
        ball=person if shared else YOLO("cv/weights/yolo11m.pt")
        rows=[]
        for frame in frames:
            result=person.track(frame,persist=True,classes=[0],tracker=str(TRACKER),
                                device="cpu",verbose=False,imgsz=960,conf=.15)[0]
            before=person.predictor.trackers[0].frame_id
            ball.predict(frame,classes=[32],device="cpu",verbose=False,imgsz=960,conf=.03)
            after=person.predictor.trackers[0].frame_id
            rows.append(dict(tracked_boxes=len(result.boxes),before_ball=before,after_ball=after))
        report["shared" if shared else "isolated"]=rows
    report["condition"]="Same four video frames, same pretrained model and tracker configuration; tracker-state diagnostic only, no labelled identity/accuracy score"
    out=Path("cv/out/training_20261011/tracker_state.json");out.write_text(json.dumps(report,indent=2))
    print(report,flush=True)


if __name__=="__main__":
    main()
