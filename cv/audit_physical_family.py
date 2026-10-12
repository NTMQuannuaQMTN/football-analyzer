"""Label-assisted camera-family diagnosis; not image-only inference."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
import torch

from cv.physical_camera import BOUNDS,camera_matrix
from cv.gsr import project


def matrix(pose):return camera_matrix(torch.from_numpy(np.asarray(pose,float))[None])[0].numpy()


def main():
    root=Path('cv/out/audit8_20261011');rows=json.loads(Path('cv/out/training_20261011/pitch_lines/data/rows.json').read_text())
    output={};torch.set_num_threads(1)
    for part in ['train','val']:
        available=[r for r in rows[part] if sum(p['pitch'] is not None and p['role'] in ('player','goalkeeper','referee') for p in r['people'])>=14]
        chosen=[available[i] for i in np.linspace(0,len(available)-1,min(30,len(available)),dtype=int)];errors=[];cases=[]
        for row in chosen:
            people=[p for p in row['people'] if p['pitch'] is not None and p['role'] in ('player','goalkeeper','referee')]
            world=np.array([p['pitch'] for p in people]);feet=np.array([[(p['box'][0]+p['box'][2])/2/1920,p['box'][3]/1080] for p in people])
            order=np.argsort(feet[:,0]+feet[:,1]);fit=order[::2];check=order[1::2]
            lower=np.array([p[0] for p in BOUNDS]);upper=np.array([p[1] for p in BOUNDS]);best=None
            target=np.clip(world[fit].mean(0),[-9.,-4.],[114.,72.])
            for focal in [1.5,4.]:
                initial=np.array([52.5,95,22,*target,focal,0.])
                result=least_squares(lambda p:((project(matrix(p),world[fit])-feet[fit])*[1920,1080]).ravel(),initial,
                    bounds=(lower+1e-6,upper-1e-6),loss='soft_l1',f_scale=3,max_nfev=200)
                cost=np.mean(np.linalg.norm(project(matrix(result.x),world[fit])-feet[fit],axis=1))
                if best is None or cost<best[0]:best=(cost,result.x)
            h=np.linalg.inv(matrix(best[1]));e=np.linalg.norm(project(h,feet[check])-world[check],axis=1);errors.extend(e.tolist())
            cases.append(dict(image=row['image'],clip=row['clip'],fit_people=len(fit),held_people=len(check),pose=best[1].tolist(),errors_m=e.tolist()))
            print(part,row['image'],float(e.mean()),flush=True)
        output[part]=dict(frames=len(cases),held_people=len(errors),mean_m=float(np.mean(errors)),median_m=float(np.median(errors)),p90_m=float(np.percentile(errors,90)),cases=cases)
    output['conditions']='Label-assisted fit to half the visible reference people, scoring the other half. Tests whether the bounded pinhole family can represent these cameras, not whether an image network can estimate it. No test game used; reference labels never enter normal inference.'
    (root/'physical_camera_family.json').write_text(json.dumps(output,indent=2));print({p:{k:v for k,v in output[p].items() if k!='cases'} for p in ['train','val']},flush=True)


if __name__=='__main__':main()
