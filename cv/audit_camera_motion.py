"""Evaluate causal camera propagation, selecting policies on validation only."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from cv.camera_motion import ground_motion,propagate
from cv.camera_quality import assess_camera,plausible
from cv.gsr import GSR,load_clip,project


def stats(values):
    x=np.array(values,float)
    return dict(n=len(x),mean_m=float(x.mean()) if len(x) else None,median_m=float(np.median(x)) if len(x) else None,p90_m=float(np.percentile(x,90)) if len(x) else None)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--part',choices=['val','test'],required=True)
    ap.add_argument('--source',default='cv/out/audit8_20261011/camera_geometric')
    ap.add_argument('--out',default='cv/out/audit8_20261011/camera_motion')
    ap.add_argument('--similarity-fallback',action='store_true',help='Experimental constrained ground-flow fallback; separate output required')
    a=ap.parse_args()
    motion=ground_motion
    if a.similarity_fallback:
        from cv.camera_similarity import similarity_fallback
        motion=similarity_fallback
        if Path(a.out).resolve()==Path('cv/out/audit8_20261011/camera_motion').resolve():
            raise ValueError('Fallback experiment must preserve the original motion evaluation')
    root=Path(a.out);root.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1)
    motion_hash=hashlib.sha256(Path('cv/camera_motion.py').read_bytes()).hexdigest()
    if a.similarity_fallback:motion_hash=hashlib.sha256((motion_hash+hashlib.sha256(Path('cv/camera_similarity.py').read_bytes()).hexdigest()).encode()).hexdigest()
    source=json.loads((Path(a.source)/f'{a.part}.json').read_text())
    if a.part=='val':configs={f'age{age}_agreement{agreement}':dict(age=age,agreement=agreement) for age in [3,6,10] for agreement in [0,5,10]}
    else:
        choice=json.loads((root/'selection.json').read_text());configs={'selected':choice['configuration']} if choice['configuration'] is not None else {}
    configs={'baseline':dict(age=0,agreement=0),**configs};errors={k:[] for k in configs};counts={k:0 for k in configs};common={k:[] for k in configs};cases=[]
    indexed={(str(r['clip']),Path(r['image']).stem.split('_')[-1]):r for r in source['cases']}
    for clip in sorted(set(r['clip'] for r in source['cases'])):
        info,frames=load_clip(GSR/'valid'/clip);cache=root/'flow'/motion_hash[:16]/clip;cache.mkdir(parents=True,exist_ok=True)
        previous=None;states={k:dict(h=None,anchor_time=-100) for k in configs}
        for i,f in enumerate(frames[::5]):
            image=cv2.imread(str(f['file']));time=i*.2
            path=cache/f'{f["file"].stem}.npz'
            if previous is None:warp=None
            elif path.exists():
                value=np.load(path);warp=value['h'] if bool(value['valid']) else None
            else:
                warp,diagnostic=motion(previous,image);np.savez_compressed(path,h=warp if warp is not None else np.eye(3),valid=warp is not None)
            previous=image
            row=indexed.get((clip,f['file'].stem));anchor=None
            quality_image=cv2.imread(row['image']) if row is not None else None
            if row is not None:
                r=row['results']['learned_image_fit']
                if 'H' in r and plausible(assess_camera(quality_image,np.array(r['H'])),.75,.25):anchor=np.array(r['H'])
            for name,cfg in configs.items():
                state=states[name];h=propagate(state['h'],warp)
                if time-state['anchor_time']>cfg['age']:h=None
                if anchor is not None:
                    accept=h is None or cfg['agreement']==0
                    if not accept:
                        points=[[.2,.4],[.5,.4],[.8,.4],[.2,.8],[.5,.8],[.8,.8]]
                        delta=np.linalg.norm(project(h,points)-project(anchor,points),axis=1)
                        accept=np.isfinite(delta).all() and np.median(delta)<=cfg['agreement']
                    if accept:h=anchor.copy();state['anchor_time']=time
                state['h']=h
                if row is None:continue
                reference=[p for p in f['people'] if p['pitch'] is not None and p['role'] in ('player','goalkeeper','referee')]
                result=dict(clip=clip,image=str(f['file']),configuration=name,supported=False,people=len(reference),anchor_age_s=time-state['anchor_time'])
                if h is not None and reference and plausible(assess_camera(quality_image,h),.75,.25):
                    feet=[[(p['box'][0]+p['box'][2])/2/1920,p['box'][3]/1080] for p in reference]
                    error=np.linalg.norm(project(h,feet)-np.array([p['pitch'] for p in reference]),axis=1)
                    if np.isfinite(error).all():
                        errors[name].extend(error.tolist());counts[name]+=1;result.update(supported=True,H=h.tolist(),errors_m=error.tolist())
                        if anchor is not None:common[name].extend(np.linalg.norm(project(anchor,feet)-np.array([p['pitch'] for p in reference]),axis=1).tolist())
                cases.append(result)
        print(a.part,clip,flush=True)
    summary={k:dict(frames=len(source['cases']),supported_frames=counts[k],coverage=counts[k]/len(source['cases']),errors=stats(errors[k]),common_anchor_errors=stats(common[k])) for k in configs}
    output=dict(part=a.part,configurations=configs,summary=summary,cases=cases,source_checkpoint=source['checkpoint'],source_checkpoint_sha256=source.get('checkpoint_sha256'),motion_source_sha256=motion_hash,
        similarity_fallback=a.similarity_fallback,
        conditions='Past/current images only. Cameras estimated at fixed image checkpoints, propagated by ground features. Exact visible feet/pitch labels score only. Rejecting views changes coverage. No detector, named identity or hidden-player score.')
    (root/f'{a.part}.json').write_text(json.dumps(output,indent=2))
    if a.part=='val':
        base=summary['baseline'];eligible=[k for k,r in summary.items() if k!='baseline' and r['coverage']>=base['coverage'] and r['errors']['mean_m'] is not None and r['errors']['mean_m']<=base['errors']['mean_m']+.1 and r['errors']['p90_m']<=base['errors']['p90_m']+.5]
        name=max(eligible,key=lambda k:summary[k]['coverage']) if eligible else None
        (root/'selection.json').write_text(json.dumps(dict(name=name,configuration=configs[name] if name else None,validation=summary[name] if name else None,
            criterion='Maximise coverage, validation mean no more than +0.1m and p90 no more than +0.5m versus unchanged image-only filter'),indent=2))
    print(summary,flush=True)


if __name__=='__main__':main()
