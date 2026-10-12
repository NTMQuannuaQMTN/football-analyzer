"""Image-only quality flags: validation selection and explicit abstention costs."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from cv.camera_quality import assess_camera,plausible


def stats(values):
    e=np.asarray(values,float)
    return dict(n=len(e),mean_m=float(e.mean()) if len(e) else None,median_m=float(np.median(e)) if len(e) else None,p90_m=float(np.percentile(e,90)) if len(e) else None)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--selection',help='Fixed validation selection to apply to test')
    a=ap.parse_args();source=json.loads(Path(a.source).read_text());models=list(source['summary']);cases=[]
    for i,c in enumerate(source['cases']):
        image=cv2.imread(c['image']);row=dict(image=c['image'],clip=c['clip'],people=c['people'],models={})
        for name in models:
            r=c['results'][name]
            row['models'][name]=dict(quality=assess_camera(image,np.array(r['H'])) if 'H' in r else dict(status='unsupported'),errors_m=r.get('errors_m'))
        cases.append(row)
        if i%100==0:print(i,len(source['cases']),flush=True)
    fixed=json.loads(Path(a.selection).read_text()) if a.selection else None
    configurations=[(fixed['model'],fixed['inside'],fixed['scale'])] if fixed else [(n,inside,scale) for n in models if n.startswith('learned') for inside in [.5,.75,.9] for scale in [.1,.25,.5]]
    options=[];total_people=sum(c['people'] for c in cases)
    for name,inside,scale in configurations:
        values=[];accepted=0;rejected_cost=0;bad_frames=0;common_baseline=[]
        for c in cases:
            r=c['models'][name]
            if r['errors_m'] is not None and plausible(r['quality'],inside,scale):
                accepted+=1;values.extend(r['errors_m']);bad_frames+=np.percentile(r['errors_m'],90)>5
                baseline=c['models'].get('learned_dlt', {}).get('errors_m')
                if baseline is not None:common_baseline.extend(baseline)
            else:rejected_cost+=20*c['people']
        options.append(dict(model=name,inside=inside,scale=scale,accepted_frames=accepted,frame_coverage=accepted/len(cases),
            position_errors=stats(values),bad_accepted_frames_p90_over_5m=int(bad_frames),common_baseline_errors=stats(common_baseline),
            selection_cost_m=(sum(min(v,20) for v in values)+rejected_cost)/max(1,total_people)))
    eligible=[r for r in options if r['frame_coverage']>=.4 and r['position_errors']['mean_m']<=3 and r['position_errors']['p90_m']<=5]
    selected=max(eligible,key=lambda r:r['frame_coverage']) if eligible else None
    if fixed:selected=options[0]
    result=dict(part=source['part'],frames=len(cases),total_labelled_people=total_people,options=options,selected=selected,cases=cases,
        conditions='Image-only quality checks; labelled feet score only. Rejection changes coverage; supported/common-set errors are not full-coverage accuracy. Operational validation target: >=40% frames, accepted mean <=3m and p90 <=5m; not a probability guarantee.')
    path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2));print('selected',selected,flush=True)


if __name__=='__main__':main()
