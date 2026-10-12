"""Paired camera-motion ablation, sharing identical raw person detections."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.trackers.bot_sort import BOTSORT

from cv.broadcast_motion import BroadcastMotion, AssociationGuardedMotion
from cv.detect_track import TRACKER
from cv.evaluate_session import iou_matrix, matched_pairs
from cv.gsr import clips, load_clip


def evaluate(part,root):
    torch.set_num_threads(1);cv2.setNumThreads(1)
    model=YOLO('cv/weights/yolo11m.pt')
    games=('3','5') if part=='val' else ('2',)
    selected=[]
    for game in games:
        selected.extend([p for p in clips('valid') if load_clip(p)[0]['game_id']==game][:1 if part=='val' else 2])
    report={k:dict(gt=0,pred=0,matched=0,id_switches=0,matched_transitions=0) for k in ('raw','legacy_motion','guarded_pitch_motion','association_guarded_motion','no_motion')}
    for clip in selected:
        _,frames=load_clip(clip); trackers={}; previous={k:{} for k in report}
        for name in ('legacy_motion','guarded_pitch_motion','association_guarded_motion','no_motion'):
            cfg=yaml.safe_load(TRACKER.read_text());cfg['device']='cpu'
            if name=='no_motion':cfg['gmc_method']='none'
            tracker=BOTSORT(SimpleNamespace(**cfg))
            if name=='guarded_pitch_motion':tracker.gmc=BroadcastMotion()
            if name=='association_guarded_motion':tracker.gmc=AssociationGuardedMotion()
            trackers[name]=tracker
        for i,f in enumerate(frames[::5]):
            image=cv2.imread(str(f['file']));cache=root/'cache'/f'{clip.name}_{f["file"].stem}.npy';cache.parent.mkdir(parents=True,exist_ok=True)
            if cache.exists():arr=np.load(cache)
            else:
                res=model.predict(image,classes=[0],imgsz=960,conf=.15,iou=.5,device='cpu',verbose=False)[0]
                arr=res.boxes.data.cpu().numpy();np.save(cache,arr)
            outputs={'raw':(arr[:,:4],None)}
            for name,tracker in trackers.items():
                tracks=tracker.update(Boxes(arr,image.shape[:2]).numpy(),image)
                outputs[name]=(tracks[:,:4],tracks[:,4]) if len(tracks) else (np.empty((0,4)),np.empty(0))
            gt=[p for p in f['people'] if p['role'] in ('player','goalkeeper','referee')]
            for name,(boxes,ids) in outputs.items():
                r=report[name];r['gt']+=len(gt);r['pred']+=len(boxes)
                matrix=iou_matrix(boxes,[p['box'] for p in gt])
                if not matrix.size:continue
                for a,b in matched_pairs(matrix):
                    r['matched']+=1
                    if ids is not None:
                        key=gt[b]['track'];identity=int(ids[a])
                        if key in previous[name]:
                            r['matched_transitions']+=1;r['id_switches']+=int(previous[name][key]!=identity)
                        previous[name][key]=identity
            if i%50==0:print(part,clip.name,i,{k:v['matched'] for k,v in report.items()},flush=True)
    for r in report.values():
        r['precision']=r['matched']/max(1,r['pred']);r['recall']=r['matched']/max(1,r['gt'])
    output=dict(part=part,clips=[p.name for p in selected],stride=5,sampling_hz=5,imgsz=960,confidence=.15,results=report,matching='threshold-first maximum-cardinality Hungarian',
                note='Same cached raw predictions for each tracker. IoU>=0.5 Hungarian matches; switches count changes between matched sightings, including gaps. Diagnostic, not full HOTA/IDF1 and not named-player accuracy.')
    (root/f'{part}.json').write_text(json.dumps(output,indent=2));print(output,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--part',choices=['val','test'],default='val');ap.add_argument('--out',default='cv/out/training_20261011/tracking')
    a=ap.parse_args();root=Path(a.out);root.mkdir(parents=True,exist_ok=True);evaluate(a.part,root)
