"""Paired raw/Kalman footpoint diagnosis on frozen detections.

Reference boxes and reference-fitted cameras are scoring-only. The camera
diagnostic deliberately excludes independent calibration and hidden players.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.utils.gmc import GMC

from cv.audit_tracking import FixedMotion
from cv.evaluate_session import iou_matrix,matched_pairs
from cv.gsr import GSR,load_clip,oracle_homography,project


def foot(box):
    return np.array([(box[0]+box[2])/2,box[3]],float)


def stats(values,unit):
    v=np.array(values,float)
    return dict(n=len(v),unit=unit,mean=float(v.mean()) if len(v) else None,
                median=float(np.median(v)) if len(v) else None,p90=float(np.percentile(v,90)) if len(v) else None)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--part',choices=['val','test'],required=True)
    ap.add_argument('--tracking',default='cv/out/audit8_20261011/tracking_generic');a=ap.parse_args()
    root=Path(a.tracking);source=json.loads((root/f'{a.part}.json').read_text())
    chosen=json.loads((root/'selection.json').read_text())
    source['configurations']={'legacy':source['configurations']['legacy'],'selected':chosen['configuration']}
    cache=next((root/'cache').iterdir());cv2.setNumThreads(1)
    keys=['raw_pixel','kalman_pixel','raw_pitch','kalman_pitch','filter_displacement_pixel']
    errors={name:{k:[] for k in keys} for name in source['configurations']};counts={name:Counter() for name in errors}
    paired=[]
    for name in source['clips']:
        clip=GSR/'valid'/name;info,frames=load_clip(clip);cv2.setRNGSeed(11026)
        motion=GMC(method='sparseOptFlow')
        trackers={k:BOTSORT(SimpleNamespace(**cfg)) for k,cfg in source['configurations'].items()}
        for tr in trackers.values():tr.gmc=FixedMotion()
        for frame in frames[::source['stride']]:
            image=cv2.imread(str(frame['file']));raw=np.load(cache/f'{name}_{frame["file"].stem}.npy')
            warp=motion.apply(image);gt=[p for p in frame['people'] if p['role'] in ('player','goalkeeper','referee')]
            h=oracle_homography(frame['people'])
            for kind,tr in trackers.items():
                tr.gmc.warp=warp;tracks=tr.update(Boxes(raw,image.shape[:2]).numpy(),image)
                boxes=tracks[:,:4] if len(tracks) else np.empty((0,4));counts[kind]['frames']+=1
                for r,c in matched_pairs(iou_matrix(boxes,[p['box'] for p in gt])):
                    index=int(tracks[r,-1]);assert 0<=index<len(raw)
                    measured=foot(raw[index,:4]);filtered=foot(boxes[r]);reference=foot(gt[c]['box'])
                    result=dict(clip=name,image=str(frame['file']),configuration=kind,track=gt[c]['track'],role=gt[c]['role'])
                    for key,value in [('raw_pixel',np.linalg.norm(measured-reference)),('kalman_pixel',np.linalg.norm(filtered-reference)),
                                      ('filter_displacement_pixel',np.linalg.norm(filtered-measured))]:
                        errors[kind][key].append(float(value));result[key]=float(value)
                    if h is not None and gt[c]['pitch'] is not None:
                        p=project(h,[measured,filtered]);e=np.linalg.norm(p-np.array(gt[c]['pitch']),axis=1)
                        if np.isfinite(e).all():
                            for key,value in zip(['raw_pitch','kalman_pitch'],e):errors[kind][key].append(float(value));result[key]=float(value)
                    paired.append(result)
        print(a.part,name,flush=True)
    result=dict(part=a.part,tracking_source=str(root/f'{a.part}.json'),model_sha256=source['model_sha256'],configurations=source['configurations'],
        results={name:{k:stats(v,'metres' if 'pitch' in k else 'pixels at 1920x1080') for k,v in rows.items()} for name,rows in errors.items()},
        rows=paired,conditions='Same IoU>=0.5 tracked/labelled pairs. Raw boxes indexed by the tracker detection index. Pitch uses reference-fitted homography; this isolates foot measurement and does not establish camera accuracy. No named or hidden player evaluation.')
    (root/f'feet_{a.part}.json').write_text(json.dumps(result,indent=2));print(result['results'],flush=True)


if __name__=='__main__':main()
