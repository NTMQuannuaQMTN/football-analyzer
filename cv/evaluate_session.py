"""Paired held-out detection/footpoint evaluation with explicit calibration conditions."""
import argparse
import json
from pathlib import Path
import hashlib

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from cv.gsr import clips,load_clip,oracle_homography,project
from backend.reconstruction.session_evaluation import summarize


def iou_matrix(a,b):
    a=np.asarray(a,float).reshape(-1,4); b=np.asarray(b,float).reshape(-1,4)
    lo=np.maximum(a[:,None,:2],b[None,:,:2]); hi=np.minimum(a[:,None,2:],b[None,:,2:])
    inter=np.maximum(hi-lo,0).prod(2)
    area_a=np.maximum(a[:,2:]-a[:,:2],0).prod(1)
    area_b=np.maximum(b[:,2:]-b[:,:2],0).prod(1)
    return inter/np.maximum(area_a[:,None]+area_b[None,:]-inter,1e-8)


def matched_pairs(iou, threshold=.5):
    """Maximise valid match count before overlap quality in crowded scenes."""
    if not iou.size:
        return []
    # Any invalid edge costs more than all valid edges combined. Rectangular
    # assignment may include invalid edges, which are then discarded.
    cost=np.where(iou>=threshold,1-iou,float(min(iou.shape)+1))
    rows,cols=linear_sum_assignment(cost)
    return [(r,c) for r,c in zip(rows,cols) if iou[r,c]>=threshold]


def evaluate(model_path,device="cpu",stride=30,imgsz=960):
    import torch
    torch.set_num_threads(1)
    from ultralytics import YOLO
    model=YOLO(str(model_path))
    ids=[k for k,v in model.names.items() if v in ("person","player","goalkeeper","referee")]
    digest=hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    cache=Path('cv/out/training_20261011/detection_cache')/f'{digest[:16]}_i{imgsz}_c015_nms05'
    cache.mkdir(parents=True,exist_ok=True)
    total=dict(frames=0,gt_people=0,predicted_people=0,matched_iou_50=0)
    errors=[]; clips_used=[]; foot_rows=[]
    for clip in clips("valid"):
        info,frames=load_clip(clip)
        if info["game_id"]!="2":
            continue
        clips_used.append(clip.name)
        for frame in frames[::stride]:
            gt=[p for p in frame["people"] if p["role"] in ("player","goalkeeper","referee")]
            im=cv2.imread(str(frame["file"]))
            cached=cache/f'{clip.name}_{frame["file"].stem}.npy'
            if cached.exists():
                boxes=np.load(cached)
            else:
                res=model.predict(im,classes=ids,conf=.15,iou=.5,imgsz=imgsz,device=device,verbose=False)[0]
                boxes=res.boxes.xyxy.cpu().numpy();np.save(cached,boxes)
            total["frames"]+=1;total["gt_people"]+=len(gt);total["predicted_people"]+=len(boxes)
            iou=iou_matrix(boxes,[p["box"] for p in gt])
            if not iou.size:
                continue
            pairs=matched_pairs(iou)
            total["matched_iou_50"]+=len(pairs)
            h=oracle_homography(frame["people"])
            if h is None:
                continue
            for r,c in pairs:
                p=gt[c]
                if p["pitch"] is None:
                    continue
                box=boxes[r]
                foot=((box[0]+box[2])/2,box[3])
                xy=project(h,[foot])[0]
                if not np.isfinite(xy).all():
                    continue
                error=float(np.linalg.norm(xy-p["pitch"]))
                errors.append(error)
                foot_rows.append(dict(clip=clip.name,image=str(frame["file"]),track=p["track"],
                                      predicted_box=box.tolist(),label_box=p["box"],error_m=error))
        print(clip.name,total,flush=True)
    return dict(model=str(model_path),test_game="2",clips=clips_used,imgsz=imgsz,stride=stride,
                confidence_threshold=.15,model_sha256=digest,matching='threshold-first maximum-cardinality Hungarian',**total,
                precision=total["matched_iou_50"]/max(1,total["predicted_people"]),
                recall=total["matched_iou_50"]/max(1,total["gt_people"]),
                oracle_calibrated_footpoint_errors=summarize(np.asarray(errors)),
                conditions="Box matching at IoU >= 0.5, exact assigned pairs. Pitch error uses label-fitted oracle homography: detector diagnostic, NOT independent end-to-end video-to-pitch accuracy.",
                rows=foot_rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True)
    ap.add_argument("--device",default="cpu")
    ap.add_argument("--stride",type=int,default=30)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    result=evaluate(a.model,a.device,a.stride)
    path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
