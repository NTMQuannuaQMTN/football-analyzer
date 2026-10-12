"""Separate annotated-marking diagnosis from image-only camera evaluation."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from cv.camera_geometry import fit_camera, consensus_camera
from cv.gsr import project
from cv.pitch_lines import PitchLineNet, heatmap_points, NAMES


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--part',choices=['val','test'],default='val')
    ap.add_argument('--device',default='cpu');ap.add_argument('--out',default='cv/out/audit8_20261011/camera')
    ap.add_argument('--checkpoint',default='cv/out/training_20261011/pitch_lines/best.pt')
    a=ap.parse_args();root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
    source_hashes={name:hashlib.sha256(Path('cv',name).read_bytes()).hexdigest() for name in ['audit_camera.py','camera_geometry.py','pitch_lines.py']}
    weights=Path(a.checkpoint)
    rows=json.loads(Path('cv/out/training_20261011/pitch_lines/data/rows.json').read_text())[a.part]
    checkpoint=torch.load(weights,weights_only=False)
    if checkpoint.get('architecture')=='resnet18':
        from cv.train_camera_audit import ResNetPitch
        net=ResNetPitch()
    else:net=PitchLineNet()
    net=net.to(a.device);net.load_state_dict(checkpoint['state']);net.eval()
    torch.set_num_threads(1);cv2.setNumThreads(1)
    cache=root/'heatmaps'/hashlib.sha256(weights.read_bytes()).hexdigest()[:16];cache.mkdir(parents=True,exist_ok=True)
    cases=[];errors={k:[] for k in ['annotated_dlt','annotated_image_fit','annotated_consensus','learned_dlt','learned_image_fit','learned_consensus']}
    for i,row in enumerate(rows):
        filename=cache/(Path(row['image']).stem+'.npz')
        if filename.exists():prob=np.load(filename)['prob'].astype(np.float32)
        else:
            image=cv2.cvtColor(cv2.imread(row['image']),cv2.COLOR_BGR2RGB)
            x=torch.from_numpy(image.transpose(2,0,1).copy()).float()[None]/255
            with torch.no_grad():prob=net(x.to(a.device)).sigmoid()[0].cpu().numpy()
            np.savez_compressed(filename,prob=prob.astype(np.float16))
        points={'annotated':{name:[[p['x'],p['y']] for p in pts] for name,pts in row['lines'].items() if name in NAMES},
                'learned':heatmap_points(prob)}
        gt=[p for p in row['people'] if p['pitch'] is not None and p['role'] in ('player','goalkeeper','referee')]
        feet=[((p['box'][0]+p['box'][2])/2/1920,p['box'][3]/1080) for p in gt]
        sample=dict(image=row['image'],clip=row['clip'],people=len(gt),results={})
        for kind,p in points.items():
            for method in ['dlt','image_fit','consensus']:
                name=kind+'_'+method
                h,d=consensus_camera(p) if method=='consensus' else fit_camera(p,method=='image_fit')
                result=dict(diagnostic=d)
                if h is not None and gt:
                    prediction=project(h,feet);e=np.linalg.norm(prediction-np.asarray([g['pitch'] for g in gt]),axis=1)
                    if np.isfinite(e).all():
                        errors[name].extend(e.tolist());result.update(H=h.tolist(),errors_m=e.tolist(),mean_m=float(e.mean()),median_m=float(np.median(e)))
                sample['results'][name]=result
        cases.append(sample)
        if i%50==0:print(a.part,i,len(rows),flush=True)
    summary={}
    for name,values in errors.items():
        e=np.asarray(values)
        summary[name]=dict(frames=len(rows),supported_frames=sum('errors_m' in s['results'][name] for s in cases),
                           people=len(e),mean_m=float(e.mean()) if len(e) else None,median_m=float(np.median(e)) if len(e) else None,
                           p90_m=float(np.percentile(e,90)) if len(e) else None)
    result=dict(part=a.part,summary=summary,cases=cases,checkpoint=str(weights),checkpoint_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),source_hashes=source_hashes,
        conditions='Annotated cameras use evaluation lines and are diagnostic only. Learned cameras use image predictions only; labelled feet/pitch positions are scoring-only. Common supported-sample comparisons must be reported separately. No detector or named identity score included.')
    (root/f'{a.part}.json').write_text(json.dumps(result,indent=2));print(summary,flush=True)


if __name__=='__main__':main()
