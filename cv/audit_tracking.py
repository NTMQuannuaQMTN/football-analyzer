"""Validation-selected tracking audit on fixed raw detections and whole games.

Labels score detections and anonymous track identity only. They never enter a
tracker. This does not evaluate named lineup identity or hidden players.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.utils.gmc import GMC
import yaml

from cv.detect_track import TRACKER
from cv.evaluate_session import iou_matrix, matched_pairs
from cv.gsr import clips, load_clip


def identity_counts(gt_counts, pred_counts, overlaps):
    """Global one-to-one trajectory mapping using simultaneous IoU>=0.5 edges.

    Every edge counts a frame, not a per-frame Hungarian decision. This follows
    identity-assignment counting; detection and identity use separate matches.
    Tracks must have at most one prediction per frame.
    """
    gs=list(gt_counts); ps=list(pred_counts)
    matrix=np.array([[overlaps.get((g,p),0) for p in ps] for g in gs],int)
    tp=0
    if matrix.size:
        a,b=linear_sum_assignment(-matrix)
        tp=int(matrix[a,b].sum())
    return dict(idtp=tp,idfn=sum(gt_counts.values())-tp,idfp=sum(pred_counts.values())-tp)


def configurations():
    base=yaml.safe_load(TRACKER.read_text())
    choices={'legacy':dict(base)}
    for high,new,fuse,match in [(.15,.15,True,.85),(.15,.2,False,.85),(.2,.2,False,.85),
                               (.3,.3,False,.85),(.2,.2,False,.95),(.3,.35,False,.95)]:
        name=f'h{high}_n{new}_f{int(fuse)}_m{match}'
        choices[name]={**base,'track_high_thresh':high,'new_track_thresh':new,'fuse_score':fuse,'match_thresh':match}
    choices['low_no_motion']={**choices['h0.2_n0.2_f0_m0.85'],'gmc_method':'none'}
    choices['low_long_buffer']={**choices['h0.2_n0.2_f0_m0.85'],'track_buffer':50}
    return choices


class FixedMotion:
    method='cached_sparse_flow'
    def __init__(self):self.warp=np.eye(2,3)
    def apply(self,*args):return self.warp.copy()
    def reset_params(self):self.warp=np.eye(2,3)


def summary(totals):
    t=dict(totals)
    t['precision']=t['matched']/max(1,t['pred'])
    t['recall']=t['matched']/max(1,t['gt'])
    t['detection_f1']=2*t['matched']/max(1,t['pred']+t['gt'])
    t['identity_f1']=2*t['idtp']/max(1,2*t['idtp']+t['idfp']+t['idfn'])
    t['switch_rate']=t['switches']/max(1,t['transitions'])
    return t


def evaluate(part, model_path, root, choices, clips_per_game=4, stride=5, device='cpu',features_for=None,tracker_class=BOTSORT,raw_cache_root=None,alternative_motion=None):
    if any(cfg.get('motion_source')=='ground' for cfg in choices.values()) and alternative_motion is None:
        raise ValueError('Ground compensation needs an explicit image-motion provider')
    cv2.setNumThreads(1);torch.set_num_threads(1)
    model=YOLO(str(model_path));person_ids=[k for k,v in model.names.items() if v in ('person','player','goalkeeper','referee')]
    digest=hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    cache=(Path(raw_cache_root) if raw_cache_root is not None else root)/'cache'/f'{digest[:16]}_{device}_{ultralytics.__version__}_960_c015_i05'
    cache.mkdir(parents=True,exist_ok=True)
    games=['3','5'] if part=='val' else ['2']
    selected=[]
    for game in games:
        candidates=[p for p in clips('valid') if load_clip(p)[0]['game_id']==game]
        # Spread through each game's available clips, fixed independently of results.
        indexes=np.linspace(0,len(candidates)-1,min(clips_per_game,len(candidates)),dtype=int)
        selected.extend(candidates[i] for i in indexes)
    totals={name:Counter(gt=0,pred=0,matched=0,switches=0,transitions=0,idtp=0,idfn=0,idfp=0) for name in choices}
    per_clip=[]
    for clip in selected:
        info,frames=load_clip(clip);cv2.setRNGSeed(11026)
        motion=GMC(method='sparseOptFlow')
        trackers={name:(tracker_class if cfg.get('with_reid') else BOTSORT)(SimpleNamespace(**cfg)) for name,cfg in choices.items()}
        for name,tr in trackers.items():
            if choices[name]['gmc_method']!='none':tr.gmc=FixedMotion()
        gt_counts=Counter();pred_counts={k:Counter() for k in choices};overlaps={k:Counter() for k in choices}
        previous={k:{} for k in choices};counts={k:Counter(gt=0,pred=0,matched=0,switches=0,transitions=0) for k in choices}
        for index,f in enumerate(frames[::stride]):
            image=cv2.imread(str(f['file']))
            path=cache/f'{clip.name}_{f["file"].stem}.npy'
            if path.exists():raw=np.load(path)
            else:
                result=model.predict(image,classes=person_ids,imgsz=960,conf=.15,iou=.5,device=device,verbose=False)[0]
                raw=result.boxes.data.cpu().numpy();np.save(path,raw)
            warp=motion.apply(image)
            other_warp=alternative_motion(image,clip.name,f['file'].stem) if alternative_motion is not None else None
            features=features_for(image,raw,clip.name,f['file'].stem) if features_for is not None else None
            gt=[p for p in f['people'] if p['role'] in ('player','goalkeeper','referee')]
            gt_counts.update(p['track'] for p in gt)
            for name,tr in trackers.items():
                if choices[name]['gmc_method']!='none':tr.gmc.warp=other_warp if choices[name].get('motion_source')=='ground' else warp
                tracks=tr.update(Boxes(raw,image.shape[:2]).numpy(),image,feats=features if choices[name].get('with_reid') else None)
                box=tracks[:,:4] if len(tracks) else np.empty((0,4))
                ids=tracks[:,4].astype(int) if len(tracks) else np.empty(0,int)
                assert len(ids)==len(set(ids.tolist())), 'Duplicate predicted ID in one frame'
                pred_counts[name].update(ids.tolist())
                c=counts[name];c['gt']+=len(gt);c['pred']+=len(box)
                matrix=iou_matrix(box,[p['box'] for p in gt])
                for a,b in np.argwhere(matrix>=.5):overlaps[name][gt[b]['track'],int(ids[a])]+=1
                for a,b in matched_pairs(matrix):
                    c['matched']+=1;key=gt[b]['track'];identity=int(ids[a])
                    if key in previous[name]:
                        c['transitions']+=1;c['switches']+=previous[name][key]!=identity
                    previous[name][key]=identity
            if index%50==0:print(part,clip.name,index,flush=True)
        rows={}
        for name in choices:
            counts[name].update(identity_counts(gt_counts,pred_counts[name],overlaps[name]))
            totals[name].update(counts[name]);rows[name]=summary(counts[name])
        per_clip.append(dict(clip=clip.name,game=info['game_id'],results=rows))
        output=dict(part=part,model=str(model_path),model_sha256=digest,device=device,ultralytics=ultralytics.__version__,
                    clips=[p.name for p in selected],stride=stride,sampling_hz=25/stride,configurations=choices,
                    results={name:summary(t) for name,t in totals.items()},per_clip=per_clip,
                    conditions='Anonymous visible tracking: identical raw predictions, IoU>=0.5; no pitch calibration, named lineup mapping or hidden-player score. Shared sparse flow is independent of detections. IDs mapped separately per clip; future information only in evaluation trajectory matching.')
        (root/f'{part}.json').write_text(json.dumps(output,indent=2))
    return output


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--part',choices=['val','test'],required=True)
    ap.add_argument('--model',default='cv/weights/yolo11m.pt');ap.add_argument('--out',default='cv/out/audit8_20261011/tracking')
    ap.add_argument('--device',default='cpu');ap.add_argument('--clips-per-game',type=int,default=4)
    a=ap.parse_args();root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
    choices=configurations()
    if a.part=='test':
        selected=json.loads((root/'selection.json').read_text())
        choices={'legacy':choices['legacy'],'selected':selected['configuration']}
    output=evaluate(a.part,a.model,root,choices,a.clips_per_game,device=a.device)
    if a.part=='val':
        baseline=output['results']['legacy']
        eligible=[name for name,r in output['results'].items() if r['precision']>=baseline['precision']-.015 and r['identity_f1']>=baseline['identity_f1']]
        name=max(eligible,key=lambda n:output['results'][n]['identity_f1'])
        (root/'selection.json').write_text(json.dumps(dict(name=name,configuration=choices[name],validation=output['results'][name],
            criterion='Highest validation identity F1, precision no more than 1.5 percentage points below legacy and identity F1 not below legacy'),indent=2))
    print({k:v for k,v in output.items() if k in ('part','results')},flush=True)


if __name__=='__main__':main()
