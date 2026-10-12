"""Private paired broadcast experiments using identical image/detection inputs.

No match export or production checkpoint is modified. Unconfirmed detections
remain anonymous image measurements; they never receive a fabricated slot ID.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.utils.gmc import GMC
import yaml

from cv.audit_tracking import FixedMotion
from cv.detect_track import TRACKER,PERSON_NAMES,BALL_NAMES,shirt_color,is_live_view
from cv.soccernet import DEFAULT_GAME,Calibration,game_dir,on_main_camera
from cv.track_namespace import TrackNamespace

ROOT=Path('cv/out/audit8_20261011/broadcast')


def prepare(root,start=1440,end=2040,fps=5):
    root.mkdir(parents=True,exist_ok=True);(root/'images').mkdir(exist_ok=True)
    video=game_dir(DEFAULT_GAME)/'1_720p.mkv';cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():raise FileNotFoundError(video)
    vfps=cap.get(cv2.CAP_PROP_FPS) or 25;step=max(1,int(round(vfps/fps)))
    cap.set(cv2.CAP_PROP_POS_MSEC,start*1000);index=int(round(start*vfps));cal=Calibration(DEFAULT_GAME,1);rows=[]
    while cap.grab():
        t=index/vfps;index+=1
        if t>end:break
        if (index-1)%step:continue
        row=dict(t=round(t,3),main_camera=on_main_camera(DEFAULT_GAME,1,t),H=None,image=None)
        if row['main_camera']:
            ok,image=cap.retrieve()
            if not ok:raise RuntimeError(f'Unreadable video at {t}')
            image=cv2.resize(image,(1280,720));name=f'{len(rows):05d}.jpg';cv2.imwrite(str(root/'images'/name),image,[cv2.IMWRITE_JPEG_QUALITY,95])
            h=cal.at(t);row.update(image=name,H=h.tolist() if h is not None else None)
        rows.append(row)
    cap.release();(root/'frames.json').write_text(json.dumps(rows,indent=2))
    (root/'source.json').write_text(json.dumps(dict(game=DEFAULT_GAME,half=1,start=start,end=end,fps=vfps/step,
        source_video=str(video),retrospective=True,camera='Existing CCBV at nearest 2Hz timestamp; no ground truth',
        camera_labels='shot-end intervals; non-main views excluded before inference',licensing='Existing local asset, private review only'),indent=2))
    print('prepared',len(rows),'samples; main images',sum(r['image'] is not None for r in rows),flush=True)


def predict(root,model_path,device):
    model=YOLO(model_path);torch.set_num_threads(1);cv2.setNumThreads(1);cv2.setRNGSeed(11026)
    person=[k for k,v in model.names.items() if v in PERSON_NAMES];ball=[k for k,v in model.names.items() if v in BALL_NAMES]
    rows=json.loads((root/'frames.json').read_text());cache=root/'raw';cache.mkdir(exist_ok=True)
    config=dict(model=model_path,model_sha256=hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),device=device,
        imgsz=1280,combined_conf=.03,person_conf=.15,nms_iou=.7,agnostic_nms=True,names=model.names,seed=11026,
        conditions='Single combined person/ball prediction, JPEG source; both paired trackers receive identical cached predictions. Not identical to historical separate predictions.')
    config_path=root/'prediction.json'
    if config_path.exists() and json.loads(config_path.read_text())!=json.loads(json.dumps(config)):
        raise ValueError('Prediction configuration changed; use a new isolated root')
    config_path.write_text(json.dumps(config,indent=2))
    for i,row in enumerate(rows):
        if row['image'] is None:continue
        path=cache/(Path(row['image']).stem+'.json')
        if path.exists():continue
        image=cv2.imread(str(root/'images'/row['image']))
        result=model.predict(image,classes=person+ball,imgsz=1280,conf=.03,iou=.7,agnostic_nms=True,device=device,verbose=False)[0]
        raw=result.boxes.data.cpu().numpy();people=raw[np.isin(raw[:,5],person)&(raw[:,4]>=.15)]
        balls=raw[np.isin(raw[:,5],ball)];balls=balls[np.argsort(-balls[:,4])[:3]]
        colors=[shirt_color(image,b) for b in people[:,:4]]
        path.write_text(json.dumps(dict(people=people.tolist(),balls=balls.tolist(),colors=colors)))
        if i%100==0:print('predicted',i,len(rows),flush=True)


def track(root,tracker_path,measurement='raw',appearance_checkpoint=None,only_appearance=False):
    rows=json.loads((root/'frames.json').read_text());source=json.loads((root/'source.json').read_text());prediction=json.loads((root/'prediction.json').read_text())
    configs={'legacy':yaml.safe_load(TRACKER.read_text()),'selected':yaml.safe_load(Path(tracker_path).read_text())}
    net=None;embedding_digest=None
    if appearance_checkpoint:
        from cv.appearance_tracker import AppearanceBOTSORT
        from cv.person_embedding import PersonEmbedding
        checkpoint=Path(appearance_checkpoint);embedding_digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        selection=json.loads(Path('cv/out/audit8_20261011/appearance_tracking/selection.json').read_text())
        if selection['checkpoint_sha256']!=embedding_digest or selection['configuration'] is None:
            raise ValueError('Appearance needs the validation-selected checkpoint/configuration')
        configs['appearance']=selection['configuration']
        net=PersonEmbedding();net.load_state_dict(torch.load(checkpoint,weights_only=False)['state']);net.eval();torch.set_num_threads(1)
    if only_appearance:
        if net is None:raise ValueError('--only-appearance requires a selected embedding checkpoint')
        configs={'appearance':configs['appearance']}
    trackers={k:(AppearanceBOTSORT if k=='appearance' else BOTSORT)(SimpleNamespace(**v)) for k,v in configs.items()};motion=GMC(method='sparseOptFlow')
    namespaces={k:TrackNamespace() for k in configs}
    for tr in trackers.values():tr.gmc=FixedMotion()
    paths={k:root/f'{k}_{measurement}' for k in configs}
    for p in paths.values():p.mkdir(exist_ok=True)
    files={k:(p/'detections.jsonl').open('w') for k,p in paths.items()};cv2.setNumThreads(1);cv2.setRNGSeed(11026)
    try:
        for i,row in enumerate(rows):
            if row['image'] is None:
                for k,tr in trackers.items():tr.reset();namespaces[k].camera(False);files[k].write(json.dumps(dict(t=row['t'],live=False,dets=[],unassigned_dets=[]))+'\n')
                motion.reset_params();continue
            image=cv2.imread(str(root/'images'/row['image']));data=json.loads((root/'raw'/(Path(row['image']).stem+'.json')).read_text())
            raw=np.array(data['people'],np.float32).reshape(-1,6);h=np.array(row['H']) if row['H'] is not None else None
            live=is_live_view(image,h,raw[:,:4]);warp=motion.apply(image)
            features=None
            if net is not None and live:
                from cv.gsr_train import crop,to_tensor
                cache=root/'embeddings'/embedding_digest[:16];cache.mkdir(parents=True,exist_ok=True)
                path=cache/(Path(row['image']).stem+'.npy')
                if path.exists():features=np.load(path)
                else:
                    features=np.zeros((len(raw),128),np.float32);patches=[];indices=[]
                    for j,b in enumerate(raw):
                        patch=crop(image,b[:4])
                        if patch is not None:patches.append(patch);indices.append(j)
                    if patches:
                        with torch.no_grad():features[indices]=net(to_tensor(np.array(patches))).numpy()
                    np.save(path,features)
            for kind,tr in trackers.items():
                namespace=namespaces[kind];namespace.camera(live)
                if not live:
                    tr.reset();files[kind].write(json.dumps(dict(t=row['t'],live=False,dets=[],unassigned_dets=[dict(id=None,cls='person',conf=float(b[4]),box=b[:4].tolist(),reason='camera unsupported') for b in raw]))+'\n');continue
                tr.gmc.warp=warp;tracks=tr.update(Boxes(raw,image.shape[:2]).numpy(),image,feats=features if kind=='appearance' else None)
                dets=[];used=set();unsupported=[]
                for track in tracks:
                    j=int(track[-1]);assert 0<=j<len(raw);used.add(j)
                    role=prediction['names'][str(int(raw[j,5]))]
                    if role=='referee':continue
                    box=raw[j,:4] if measurement=='raw' else track[:4]
                    foot=[(box[0]+box[2])/2,box[3]];point=Calibration.project(h,[foot])[0]
                    if not np.isfinite(point).all() or not (-3<point[0]<108 and -3<point[1]<71):
                        unsupported.append(dict(id=None,track_id=int(track[4]),cls='person',box=raw[j,:4].tolist(),reason='pitch projection unsupported'));continue
                    dets.append(dict(id=namespace.identity(int(track[4])),native_id=int(track[4]),shot_id=namespace.shot,cls='person',role='goalkeeper' if role=='goalkeeper' else 'player',conf=round(float(raw[j,4]),3),
                        box=box.tolist(),raw_box=raw[j,:4].tolist(),filtered_box=track[:4].tolist(),
                        px=round(float(point[0]),2),py=round(float(point[1]),2),colors=data['colors'][j],measurement=measurement))
                for ball in data['balls']:
                    box=ball[:4];foot=[(box[0]+box[2])/2,(box[1]+box[3])/2];point=Calibration.project(h,[foot])[0]
                    if np.isfinite(point).all() and -3<point[0]<108 and -3<point[1]<71:
                        dets.append(dict(id=-1,cls='ball',role='ball',conf=round(ball[4],3),box=box,px=round(float(point[0]),2),py=round(float(point[1]),2)))
                unknown=unsupported+[dict(id=None,cls='person',box=raw[j,:4].tolist(),conf=float(raw[j,4]),reason='track not confirmed') for j in range(len(raw)) if j not in used]
                files[kind].write(json.dumps(dict(t=row['t'],live=True,dets=dets,unassigned_dets=unknown))+'\n')
            if i%200==0:print('tracked',i,len(rows),measurement,flush=True)
    finally:
        for f in files.values():f.close()
    for k,p in paths.items():
        meta={**source,**prediction,'tracker':configs[k],'tracker_kind':k,'measurement':measurement,'colour_seed':11026,'anonymous_tracks':True,
              'track_id_namespace':'shot-qualified-v1','embedding_checkpoint':appearance_checkpoint if k=='appearance' else None,
              'embedding_sha256':embedding_digest if k=='appearance' else None,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (p/'run.json').write_text(json.dumps(meta,indent=2))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','predict','track'])
    ap.add_argument('--root',default=str(ROOT));ap.add_argument('--model',default='cv/weights/yolo11m.pt');ap.add_argument('--device',default='mps')
    ap.add_argument('--tracker',default='cv/out/audit8_20261011/selected_tracker.yaml');ap.add_argument('--measurement',choices=['raw','kalman'],default='raw')
    ap.add_argument('--appearance-checkpoint');ap.add_argument('--only-appearance',action='store_true')
    a=ap.parse_args();root=Path(a.root)
    if a.stage=='prepare':prepare(root)
    elif a.stage=='predict':predict(root,a.model,a.device)
    else:track(root,a.tracker,a.measurement,a.appearance_checkpoint,a.only_appearance)


if __name__=='__main__':main()
