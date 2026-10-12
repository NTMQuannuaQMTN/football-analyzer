"""Evaluate existing paint refinement with current image/detections only."""
import argparse
import hashlib
import html
import json
from pathlib import Path

import cv2
import numpy as np
import ultralytics

from cv.camera_quality import assess_camera,plausible
from cv.gsr import GSR,load_clip,project
from cv.refine_calibration import refine
from cv.session_report import table


ROOT=Path('cv/out/audit8_20261011')


def stats(values):
    x=np.array(values,float)
    return dict(n=len(x),mean_m=float(x.mean()) if len(x) else None,median_m=float(np.median(x)) if len(x) else None,p90_m=float(np.percentile(x,90)) if len(x) else None)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--part',choices=['val','test'],required=True);args=parser.parse_args()
    root=ROOT/'paint_refinement';root.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1)
    method_hash=hashlib.sha256(Path('cv/refine_calibration.py').read_bytes()).hexdigest()
    if args.part=='test':
        selection=json.loads((root/'selection.json').read_text())
        if not selection['eligible'] or selection['method_sha256']!=method_hash:raise ValueError('No unchanged eligible validation selection')
    source=json.loads((ROOT/'camera_motion'/f'{args.part}.json').read_text())
    config=json.loads((ROOT/'camera_motion/selection.json').read_text())['name'] if args.part=='val' else 'selected'
    digest=hashlib.sha256(Path('cv/weights/yolo11m.pt').read_bytes()).hexdigest()
    raw_root=ROOT/'tracking_generic/cache'/f'{digest[:16]}_cpu_{ultralytics.__version__}_960_c015_i05'
    cases=[c for c in source['cases'] if c['configuration']==config]
    lookups={};errors={'baseline':[],'refined':[]};changed={'baseline':[],'refined':[]};results=[];accepted=0
    for index,case in enumerate(cases):
        result=dict(case);result['paint_refinement']=dict(accepted=False,reason='baseline unsupported')
        if case['supported']:
            clip=case['clip'];stem=Path(case['image']).stem
            if clip not in lookups:
                _,frames=load_clip(GSR/'valid'/clip);lookups[clip]={f['file'].stem:f for f in frames}
            frame=lookups[clip][stem];image=cv2.imread(str(frame['file']));height,width=image.shape[:2]
            h=np.array(case['H']);pixel_h=h@np.diag([1/width,1/height,1.])
            cache=raw_root/f'{clip}_{stem}.npy'
            if not cache.is_file():raise FileNotFoundError('Missing current-frame predicted boxes: '+str(cache))
            raw=np.load(cache)
            candidate,diagnostic=refine(image,pixel_h,raw[:,:4])
            candidate=candidate@np.diag([width,height,1.])
            if diagnostic['accepted'] and plausible(assess_camera(image,candidate),.75,.25):
                h=candidate;accepted+=1
            elif diagnostic['accepted']:
                diagnostic.update(accepted=False,reason='refinement failed unchanged physical plausibility gate')
            # Reference data is read only after the candidate camera is fixed.
            people=[p for p in frame['people'] if p['pitch'] is not None and p['role'] in ('player','goalkeeper','referee')]
            feet=[[(p['box'][0]+p['box'][2])/2/width,p['box'][3]/height] for p in people]
            error=np.linalg.norm(project(h,feet)-np.array([p['pitch'] for p in people]),axis=1)
            if not np.isfinite(error).all():raise ValueError('Non-finite corrected camera error')
            base=case['errors_m'];assert len(base)==len(error)
            errors['baseline'].extend(base);errors['refined'].extend(error.tolist())
            if diagnostic['accepted']:
                changed['baseline'].extend(base);changed['refined'].extend(error.tolist())
            result.update(H=h.tolist(),errors_m=error.tolist(),paint_refinement=diagnostic)
        results.append(result)
        if index%100==0:print(args.part,index,len(cases),accepted,flush=True)
    summary={k:stats(v) for k,v in errors.items()}
    output=dict(part=args.part,cases=results,summary=summary,changed_subset={k:stats(v) for k,v in changed.items()},
        supported_frames=sum(c['supported'] for c in cases),total_frames=len(cases),accepted_refinements=accepted,method_sha256=method_hash,
        conditions='Existing fixed paint-refinement method, current image and cached stock-detector boxes only; future images, labelled feet and named identity are not inputs. Same supported cameras/people as frozen propagated baseline. Failed image/physical checks fall back to the original camera. Labelled visible feet score only. No hidden-player or EPL accuracy claim.')
    (root/f'{args.part}.json').write_text(json.dumps(output,indent=2))
    if args.part=='val':
        eligible=summary['refined']['mean_m']<=summary['baseline']['mean_m']-.1 and summary['refined']['p90_m']<=summary['baseline']['p90_m']
        (root/'selection.json').write_text(json.dumps(dict(eligible=eligible,method_sha256=method_hash,
            criterion='At least 0.1m paired validation mean reduction with non-increasing p90; exactly preserved support'),indent=2))
    rows=[[name,v['n'],round(v['mean_m'],3),round(v['median_m'],3),round(v['p90_m'],3)] for name,v in summary.items()]
    (root/f'{args.part}.html').write_text('<!doctype html><meta charset="utf-8"><title>Current-image camera refinement</title><style>body{font:16px system-ui;margin:30px}table{border-collapse:collapse}th,td{border:1px solid #aaa;padding:9px}</style><h1>Current-image paint refinement: '+args.part+'</h1><p>'+html.escape(output['conditions'])+'</p>'+table(['Method','People','Mean m','Median m','P90 m'],rows)+f'<p>{accepted} accepted refinements of {output["supported_frames"]} supported frames; {len(cases)} total timestamps.</p>')
    print(summary,'accepted',accepted,flush=True)


if __name__=='__main__':main()
