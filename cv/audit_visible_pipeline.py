"""Combine real detections/tracking with independently estimated pitch cameras.

Reference labels only score IoU pairs and position errors. The explicitly named
oracle control fits labels and is excluded from deployable accuracy claims.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.utils.gmc import GMC

from cv.audit_tracking import FixedMotion
from cv.evaluate_session import iou_matrix,matched_pairs
from cv.gsr import GSR,load_clip,oracle_homography,project


def stats(values):
    e=np.array(values,float)
    return dict(n=len(e),mean_m=float(e.mean()) if len(e) else None,median_m=float(np.median(e)) if len(e) else None,p90_m=float(np.percentile(e,90)) if len(e) else None)


def dense_camera_cases(dense,part):
    if dense['part']!=part or dense.get('camera_H_coordinates')!='normalised_0_to_1_image_to_pitch_metres' or dense.get('sampling_hz')!=5:
        raise ValueError('Dense camera split, cadence or coordinate units do not match')
    cases=[];seen=set()
    for row in dense['cases']:
        key=(row['clip'],row['frame'])
        if key in seen:raise ValueError('Duplicate dense camera frame')
        seen.add(key)
        for name,configuration in [('strict','baseline'),('foreground_exclusion','foreground_exclusion')]:
            value=row['methods'][name];h=value['H']
            if value['camera_supported']:
                matrix=np.asarray(h,float)
                if matrix.shape!=(3,3) or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix)!=3:raise ValueError('Supported dense camera needs a finite invertible matrix')
            elif h is not None:raise ValueError('Unsupported dense camera must abstain explicitly')
            cases.append(dict(clip=row['clip'],image=str(GSR/'valid'/row['clip']/'img1'/f"{row['frame']}.jpg"),configuration=configuration,supported=value['camera_supported'],H=h))
    return cases


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--part',choices=['val','test'],required=True)
    ap.add_argument('--model',default='cv/weights/yolo11m.pt');ap.add_argument('--tracking-root',default='cv/out/audit8_20261011/tracking_generic')
    ap.add_argument('--out',default='cv/out/audit8_20261011/visible_pipeline')
    ap.add_argument('--dense-camera-source',help='Frozen all-frame camera experiment with normalized-image units')
    a=ap.parse_args()
    root=Path(a.out);root.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);cv2.setNumThreads(1)
    source=json.loads(Path(f'cv/out/audit8_20261011/camera_motion/{a.part}.json').read_text())
    selected=json.loads(Path('cv/out/audit8_20261011/camera_motion/selection.json').read_text())['name'] if a.part=='val' else 'selected'
    if a.dense_camera_source:
        dense=json.loads(Path(a.dense_camera_source).read_text())
        selected='foreground_exclusion';source={'cases':dense_camera_cases(dense,a.part)}
    cameras={(r['clip'],Path(r['image']).stem,r['configuration']):r for r in source['cases']}
    trackroot=Path(a.tracking_root);legacy=json.loads((trackroot/'val.json').read_text())['configurations']['legacy'];chosen=json.loads((trackroot/'selection.json').read_text())['configuration']
    configs={'legacy':legacy,'selected':chosen};weights=Path(a.model);digest=hashlib.sha256(weights.read_bytes()).hexdigest()
    cache=trackroot/'cache'/f'{digest[:16]}_cpu_{ultralytics.__version__}_960_c015_i05';cache.mkdir(parents=True,exist_ok=True)
    model=YOLO(str(weights));person_ids=[k for k,v in model.names.items() if v in ('person','player','goalkeeper','referee')];totals={};errors={};rows=[];frame_counts=Counter()
    for clip in sorted(set(r['clip'] for r in source['cases'])):
        _,frames=load_clip(GSR/'valid'/clip);motion=GMC(method='sparseOptFlow');cv2.setRNGSeed(11026)
        trackers={k:BOTSORT(SimpleNamespace(**v)) for k,v in configs.items()}
        for tr in trackers.values():tr.gmc=FixedMotion()
        for f in frames[::5]:
            image=cv2.imread(str(f['file']));path=cache/f'{clip}_{f["file"].stem}.npy'
            if path.exists():raw=np.load(path)
            else:
                result=model.predict(image,classes=person_ids,imgsz=960,conf=.15,iou=.5,device='cpu',verbose=False)[0]
                raw=result.boxes.data.cpu().numpy();np.save(path,raw)
            warp=motion.apply(image);tracked={}
            for k,tr in trackers.items():tr.gmc.warp=warp;tracked[k]=tr.update(Boxes(raw,image.shape[:2]).numpy(),image)
            if (clip,f['file'].stem,'baseline') not in cameras:continue
            gt=[p for p in f['people'] if p['role'] in ('player','goalkeeper','referee')]
            camera_variants={'learned_static':cameras[clip,f['file'].stem,'baseline'],'learned_propagated':cameras[clip,f['file'].stem,selected]}
            if a.dense_camera_source:camera_variants={'dense_strict':cameras[clip,f['file'].stem,'baseline'],'dense_foreground':cameras[clip,f['file'].stem,selected]}
            for name,r in camera_variants.items():frame_counts[name+'_total']+=1;frame_counts[name+'_supported']+=r['supported']
            h=oracle_homography(f['people']);camera_variants['oracle_diagnostic']=dict(supported=h is not None,H=(h@np.diag([1920,1080,1])).tolist() if h is not None else None)
            for detector,tracks in {'raw_detection':None,**tracked}.items():
                boxes=raw[:,:4] if tracks is None else tracks[:,:4] if len(tracks) else np.empty((0,4))
                pairs=matched_pairs(iou_matrix(boxes,[p['box'] for p in gt]))
                for camera,r in camera_variants.items():
                    for measurement in (['raw'] if tracks is None else ['raw','kalman']):
                        key=f'{detector}/{camera}/{measurement}';c=totals.setdefault(key,Counter());values=errors.setdefault(key,[])
                        c['gt_total']+=len(gt);c['predicted_total']+=len(boxes);c['matched_total']+=len(pairs)
                        if not r['supported']:continue
                        c['gt_camera_supported']+=len(gt);c['predicted_camera_supported']+=len(boxes);c['matched_camera_supported']+=len(pairs)
                        matrix=np.array(r['H'])
                        for i,j in pairs:
                            if gt[j]['pitch'] is None:continue
                            b=raw[int(tracks[i,-1]),:4] if tracks is not None and measurement=='raw' else boxes[i]
                            p=project(matrix,[[(b[0]+b[2])/2/1920,b[3]/1080]])[0]
                            error=float(np.linalg.norm(p-np.array(gt[j]['pitch'])))
                            if not np.isfinite(error):c['nonfinite']+=1;continue
                            values.append(error);c['matched_within_5m']+=error<=5
                            rows.append(dict(clip=clip,image=str(f['file']),variant=key,gt_track=gt[j]['track'],role=gt[j]['role'],error_m=error))
        print(a.part,clip,flush=True)
    summary={key:{**c,'position_errors_on_matched_supported_people':stats(errors[key]),
        'within_5m_recall_all_labelled_people':c['matched_within_5m']/max(1,c['gt_total']),
        'within_5m_precision_supported_camera':c['matched_within_5m']/max(1,c['predicted_camera_supported'])} for key,c in totals.items()}
    output=dict(part=a.part,model=str(weights),model_sha256=digest,tracking_root=str(trackroot),frames=dict(frame_counts),results=summary,rows=rows,
        dense_camera_source=a.dense_camera_source,dense_camera_sha256=hashlib.sha256(Path(a.dense_camera_source).read_bytes()).hexdigest() if a.dense_camera_source else None,
        conditions='Full fixed camera sample list, same cached detections per model. IoU>=0.5 matches precede pitch error calculation. Missing camera/detection lowers all-labelled <=5m recall and is excluded explicitly from conditional metre errors. Anonymous tracks only; named and hidden identities not tested. Oracle control uses labelled feet and cannot establish deployable accuracy.')
    (root/f'{a.part}.json').write_text(json.dumps(output,indent=2));print(summary,flush=True)


if __name__=='__main__':main()
