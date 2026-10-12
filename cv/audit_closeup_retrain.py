"""Frozen paired camera checks and evaluation-only user landmarks."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from cv.audit_broadcast_cameras import overlay
from cv.evaluate_manual_camera import read_labels,score
from cv.gsr import project
from cv.pitch_lines import heatmap_points,homography_from_lines
from cv.session_report import table
from cv.soccernet import DEFAULT_GAME,on_main_camera
from cv.train_camera_audit import ResNetPitch


def predict(net,image_path,device):
    """Only the current image enters camera prediction, never reference labels."""
    image=cv2.imread(str(image_path))
    if image is None:raise FileNotFoundError(image_path)
    small=cv2.resize(image,(512,288))
    tensor=torch.from_numpy(cv2.cvtColor(small,cv2.COLOR_BGR2RGB).transpose(2,0,1).copy()).float()[None]/255
    with torch.no_grad():prob=net(tensor.to(device)).sigmoid()[0].cpu().numpy()
    return homography_from_lines(heatmap_points(prob))


def stats(values):
    v=np.asarray(values,float)
    return dict(n=len(v),mean_m=float(v.mean()) if len(v) else None,
        median_m=float(np.median(v)) if len(v) else None,p90_m=float(np.percentile(v,90)) if len(v) else None)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--device',default='cpu');a=ap.parse_args()
    if not (a.root/'finished.json').is_file():raise ValueError('Freeze completed training before evaluation')
    out=a.root/'paired_evaluation';out.mkdir(exist_ok=True);torch.set_num_threads(1);cv2.setNumThreads(1)
    initial='cv/out/audit8_20261011/camera_resnet_clean/best.pt'
    nets={}
    for name,path in [('initial',initial),('selected',a.root/'camera/best.pt')]:
        net=ResNetPitch().to(a.device);net.load_state_dict(torch.load(path,weights_only=False)['state']);net.eval();nets[name]=net
    rows=json.loads(Path('cv/out/audit8_20261011/camera_training/rows.json').read_text())['val'][::4]
    per_case={name:{} for name in nets};costs={name:[] for name in nets}
    for i,row in enumerate(rows):
        predictions={name:predict(net,row['image'],a.device) for name,net in nets.items()}
        # References are first read after both image-only predictions.
        people=[p for p in row['people'] if p['pitch'] is not None and p['role'] in ('player','goalkeeper','referee')]
        feet=np.asarray([[(p['box'][0]+p['box'][2])/2/1920,p['box'][3]/1080] for p in people])
        for name,h in predictions.items():
            errors=None
            if h is not None and len(people):
                value=np.linalg.norm(project(h,feet)-np.asarray([p['pitch'] for p in people]),axis=1)
                if np.isfinite(value).all():errors=value.tolist()
            per_case[name][str(i)]=errors
            costs[name].extend(np.minimum(errors,20).tolist() if errors is not None else [20.]*len(people))
        if i%50==0:print('paired validation',i,len(rows),flush=True)
    common=[key for key in per_case['initial'] if per_case['initial'][key] is not None and per_case['selected'][key] is not None]
    summary={}
    for name in nets:
        values=[v for row in per_case[name].values() if row is not None for v in row]
        aggregate=stats(values);coverage=sum(row is not None for row in per_case[name].values())/len(rows)
        passed=coverage>=.4 and aggregate['mean_m'] is not None and aggregate['mean_m']<=3 and aggregate['p90_m']<=5
        summary[name]=dict(supported=aggregate,total_frames=len(rows),supported_frames=round(coverage*len(rows)),
            common_supported=stats([v for key in common for v in per_case[name][key]]),
            selection_cost_m=float(np.mean(costs[name])),gate_passed=passed)
    manual=Path('cv/out/audit8_20261011/manual_review');frames=json.loads((manual/'frames.json').read_text())
    references=read_labels(manual/'arsenal_manual_review.csv',frames);landmarks={name:{} for name in nets}
    for frame in frames:
        if not frame['main_camera']:continue
        image_path=manual/frame['image']
        for name,net in nets.items():
            h=predict(net,image_path,a.device);landmarks[name][frame['video_s']]=None if h is None else h@np.diag([1/1280.,1/720.,1.])
            if frame['video_s'] in (1444.2,1444.4,1810):
                cv2.imwrite(str(out/f'{name}_{frame["id"]}.jpg'),overlay(cv2.imread(str(image_path)),h,True,f'{name} image-only camera; no reference fitting'))
    manual_results={name:score(references,lambda t:matrices.get(t),lambda t:on_main_camera(DEFAULT_GAME,1,t)) for name,matrices in landmarks.items()}
    for name,r in manual_results.items():
        r['conditions']=f'Evaluation-only user landmarks scored against the frozen {name} image-only learned-line camera, not CCBV. No reference fitting or hidden-player score. Missing camera solutions remain unscored and must be counted.'
    result=dict(summary=summary,common_frames=len(common),per_case=per_case,manual=manual_results,
        conditions='Fixed identical quarter-validation images, raw learned-line DLT only. Visible reference feet and user landmarks enter scoring after prediction. No manual fitting or hidden/named-player accuracy. Selected checkpoint frozen by validation before manual diagnostics.',
        test_status='not run: selected camera validation gate failed' if not summary['selected']['gate_passed'] else 'not run: full validation is required before a test or deployment decision',promoted=False,
        checkpoints_sha256={'initial':hashlib.sha256(Path(initial).read_bytes()).hexdigest(),'selected':hashlib.sha256((a.root/'camera/best.pt').read_bytes()).hexdigest()},
        manual_csv_sha256=hashlib.sha256((manual/'arsenal_manual_review.csv').read_bytes()).hexdigest(),
        rows_sha256=hashlib.sha256(Path('cv/out/audit8_20261011/camera_training/rows.json').read_bytes()).hexdigest())
    (out/'results.json').write_text(json.dumps(result,indent=2))
    def display(value):return 'N/A' if value is None else round(value,3)
    lines=[]
    for name,s in summary.items():
        for group in ('supported','common_supported'):
            r=s[group];lines.append([name,group,s['supported_frames'],s['total_frames'],r['n'],display(r['mean_m']),display(r['median_m']),display(r['p90_m']),s['gate_passed']])
    manual_rows=[[name,r['landmark_labels'],r['scored_landmarks'],display(r['mean_landmark_error_m']),display(r['median_landmark_error_m']),display(r['p90_landmark_error_m'])] for name,r in manual_results.items()]
    (out/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Paired camera validation</title><style>body{font:16px system-ui;margin:24px}table{border-collapse:collapse;margin:15px 0}td,th{border:1px solid #aaa;padding:8px}</style><h1>Paired camera validation</h1><p>'+result['conditions']+'</p>'+table(['Camera','Condition','Supported frames','Total frames','People N','Mean m','Median m','P90 m','Camera gate'],lines)+'<h2>User landmark diagnostic</h2>'+table(['Camera','Labelled','Scored','Mean m','Median m','P90 m'],manual_rows)+'<p>'+result['test_status']+'. No deployment.</p>')
    print(summary,flush=True);print({name:{k:v for k,v in r.items() if k not in ('rows','identity_labels')} for name,r in manual_results.items()},flush=True)


if __name__=='__main__':main()
