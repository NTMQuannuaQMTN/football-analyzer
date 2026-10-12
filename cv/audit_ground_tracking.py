"""Compare foreground-independent camera compensation on frozen detections."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from cv.audit_tracking import evaluate
from cv.ground_tracker_motion import affine_from_ground


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--part',choices=['val','test'],required=True);args=parser.parse_args()
    root=Path('cv/out/audit8_20261011/ground_tracking');root.mkdir(parents=True,exist_ok=True)
    source=Path('cv/ground_tracker_motion.py');digest=hashlib.sha256(source.read_bytes()+Path('cv/camera_motion.py').read_bytes()).hexdigest()
    base=json.loads(Path('cv/out/audit8_20261011/tracking_generic/selection.json').read_text())['configuration']
    choices={'baseline':base}
    if args.part=='val':choices.update(ground={**base,'motion_source':'ground'},no_motion={**base,'gmc_method':'none'})
    else:
        selection=json.loads((root/'selection.json').read_text())
        if selection['motion_sha256']!=digest:raise ValueError('Motion source changed after validation')
        if selection['configuration'] is None:raise ValueError('No eligible ground-motion validation candidate')
        choices['selected']=selection['configuration']
    previous={'clip':None,'image':None};cache=root/'motion'/digest[:16];cache.mkdir(parents=True,exist_ok=True);diagnostics=[]
    def motion(image,clip,stem):
        path=cache/f'{clip}_{stem}.npz'
        if previous['clip']!=clip or previous['image'] is None:warp=np.eye(2,3);d=dict(status='first frame')
        elif path.exists():value=np.load(path,allow_pickle=True);warp=value['warp'];d=value['diagnostic'].item()
        else:
            warp,d=affine_from_ground(previous['image'],image);np.savez_compressed(path,warp=warp,diagnostic=np.array(d,dtype=object))
        previous.update(clip=clip,image=image);diagnostics.append(dict(clip=clip,frame=stem,**d));return warp
    started=time.perf_counter()
    result=evaluate(args.part,'cv/weights/yolo11m.pt',root,choices,4 if args.part=='val' else 18,raw_cache_root='cv/out/audit8_20261011/tracking_generic',alternative_motion=motion)
    result.update(motion_sha256=digest,motion_diagnostics=diagnostics,wall_s=time.perf_counter()-started,
        motion_conditions='Past/current ground-paint optical flow only, conservative affine approximation, identity warp on unsupported motion. Same cached detections, no player truth or future images in compensation. No named identity or hidden-position claim.')
    (root/f'{args.part}.json').write_text(json.dumps(result,indent=2))
    if args.part=='val':
        b=result['results']['baseline'];eligible=[k for k,r in result['results'].items() if k!='baseline' and r['identity_f1']>=b['identity_f1']+.005 and r['precision']>=b['precision']-.005 and r['recall']>=b['recall']-.02]
        name=max(eligible,key=lambda k:result['results'][k]['identity_f1']) if eligible else None
        (root/'selection.json').write_text(json.dumps(dict(name=name,configuration=choices[name] if name else None,motion_sha256=digest,
            criterion='At least 0.5pp validation identity F1 gain, precision loss <=0.5pp and recall loss <=2pp'),indent=2))
    print(result['results'],flush=True)


if __name__=='__main__':main()
