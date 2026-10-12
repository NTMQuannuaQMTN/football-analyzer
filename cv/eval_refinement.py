"""Controlled calibration perturbation test; not real-video absolute accuracy."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from cv.gsr import oracle_homography,project
from cv.refine_calibration import refine
from backend.reconstruction.session_evaluation import summarize


def evaluate(rows,seed=11026,every=5):
    rng=np.random.default_rng(seed); before=[];after=[];accepted=0;count=0
    clips={}
    for row in rows[::every]:
        people=row['people']; h=oracle_homography(people)
        if h is None:
            continue
        im=cv2.imread(row['image']);height,width=im.shape[:2]
        h=h@np.diag([1920/width,1080/height,1])
        delta=np.array([[1+rng.uniform(-.02,.02),0,rng.uniform(-.025,.025)],
                        [0,1+rng.uniform(-.02,.02),rng.uniform(-.025,.025)],[0,0,1]])
        delta=np.diag([width,height,1])@delta@np.diag([1/width,1/height,1])
        prior=h@np.linalg.inv(delta)
        result,diagnostic=refine(im,prior)
        gt=[p for p in people if p['pitch'] is not None and p['role']!='ball']
        feet=[((p['box'][0]+p['box'][2])/2*width/1920,p['box'][3]*height/1080) for p in gt]
        reference=np.array([p['pitch'] for p in gt])
        e0=np.linalg.norm(project(prior,feet)-reference,axis=1)
        e1=np.linalg.norm(project(result,feet)-reference,axis=1)
        before.extend(e0);after.extend(e1);count+=1;accepted+=diagnostic['accepted']
        clips.setdefault(row['clip'],{'before':[],'after':[]})
        clips[row['clip']]['before'].extend(e0);clips[row['clip']]['after'].extend(e1)
    return dict(frames=count,accepted_frames=accepted,before=summarize(before),after=summarize(after),
                by_clip={k:{a:summarize(b) for a,b in v.items()} for k,v in clips.items()},
                condition='Known label-fitted camera perturbed by up to 2.5% image translation and 2% scale. Refiner sees only image and perturbed prior. Tests robustness to simulated calibration error, not absolute accuracy on Arsenal.')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--part',choices=['val','test'],default='val');a=ap.parse_args()
    root=Path('cv/out/training_20261011')
    rows=json.loads((root/'pitch_lines/data/rows.json').read_text())[a.part]
    report=evaluate(rows)
    (root/f'refinement_{a.part}.json').write_text(json.dumps(report,indent=2))
    print({k:v for k,v in report.items() if k!='by_clip'},flush=True)


if __name__=='__main__':
    main()
