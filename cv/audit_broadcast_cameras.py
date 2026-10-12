"""Private camera-paint overlays on existing Arsenal review frames.

No labelled positions are available here: plausibility/alignment is not a
measured player-accuracy score. Keep all rendered broadcast images local.
"""
import argparse
import hashlib
from html import escape
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from cv.camera_geometry import fit_camera,consensus_camera
from cv.camera_quality import assess_camera,plausible
from cv.pitch_lines import PitchLineNet,heatmap_points
from cv.refine_calibration import pitch_markings
from cv.soccernet import Calibration,DEFAULT_GAME


def overlay(image,h,normalised,label):
    view=image.copy()
    if h is not None:
        scale=np.diag([1280,720,1]) if normalised else np.eye(3)
        world=pitch_markings(step=.75);hp=np.column_stack([world,np.ones(len(world))])@(scale@np.linalg.inv(h)).T
        with np.errstate(divide='ignore',invalid='ignore'):pixel=hp[:,:2]/hp[:,2:]
        for p in pixel[np.isfinite(pixel).all(1)&(pixel[:,0]>=0)&(pixel[:,0]<1280)&(pixel[:,1]>=0)&(pixel[:,1]<720)]:cv2.circle(view,tuple(p.astype(int)),2,(50,230,255),-1)
    cv2.rectangle(view,(0,0),(1280,32),(20,20,20),-1);cv2.putText(view,label,(12,23),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
    return view


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',default='cv/out/training_20261011/pitch_lines/best.pt')
    ap.add_argument('--out',default='cv/out/audit8_20261011/broadcast_cameras');a=ap.parse_args()
    root=Path(a.out);root.mkdir(parents=True,exist_ok=True);weights=Path(a.checkpoint)
    ck=torch.load(weights,weights_only=False)
    if ck.get('architecture')=='resnet18':
        from cv.train_camera_audit import ResNetPitch
        net=ResNetPitch()
    else:net=PitchLineNet()
    net.load_state_dict(ck['state']);net.eval();torch.set_num_threads(1);cv2.setNumThreads(1)
    source=Path('cv/out/audit8_20261011/manual_review');frames=json.loads((source/'frames.json').read_text());camera=Calibration(DEFAULT_GAME,1);cases=[];html=[]
    for frame in frames:
        if not frame['main_camera']:continue
        image=cv2.imread(str(source/frame['image']));small=cv2.resize(image,(512,288))
        x=torch.from_numpy(cv2.cvtColor(small,cv2.COLOR_BGR2RGB).transpose(2,0,1).copy()).float()[None]/255
        with torch.no_grad():prob=net(x).sigmoid()[0].numpy()
        points=heatmap_points(prob);case=dict(frame_id=frame['id'],video_s=frame['video_s'],cameras={});names=[]
        cameras={'CCBV':(camera.at(frame['video_s']),False)}
        for name,method in [('Learned DLT','dlt'),('Learned image fit','image_fit'),('Learned segment consensus','consensus')]:
            h,diagnostic=consensus_camera(points) if method=='consensus' else fit_camera(points,method=='image_fit')
            cameras[name]=(h,True)
        for name,(h,normalised) in cameras.items():
            hn=h if normalised else h@np.diag([1280,720,1]) if h is not None else None
            quality=assess_camera(small,hn);status='plausibility checks passed' if plausible(quality,.75,.25) else 'N/A / plausibility rejected'
            case['cameras'][name]=dict(H=h.tolist() if h is not None else None,normalised_image=normalised,quality=quality,plausible=plausible(quality,.75,.25))
            filename=f'{frame["id"]}_{name.lower().replace(" ","_")}.jpg';cv2.imwrite(str(root/filename),overlay(image,h,normalised,f'{name}: {status}; accuracy unverified'))
            names.append(f'<figure><img src="{filename}" width="640"><figcaption>{escape(name)} · {escape(status)}</figcaption></figure>')
        html.append(f'<h2>Video {frame["video_s"]} s</h2><div class="grid">'+''.join(names)+'</div>');cases.append(case)
    result=dict(checkpoint=str(weights),checkpoint_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),cases=cases,
        conditions='Existing private broadcast frames. Ground truth absent; image plausibility is not player accuracy or calibrated confidence. Camera coordinates/orientation need independent manual landmarks. No source exports modified.')
    (root/'cameras.json').write_text(json.dumps(result,indent=2))
    (root/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Private camera comparison</title><style>body{font:16px system-ui;margin:24px;background:#f7f7fa}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr))}figure{margin:8px}img{width:100%;height:auto}</style><h1>Private camera comparison</h1><p>'+escape(result['conditions'])+'</p>'+''.join(html));print(root/'review.html',flush=True)


if __name__=='__main__':main()
