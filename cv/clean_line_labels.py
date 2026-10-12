"""Training-only semantic line cleanup from independent labelled visible feet.

Never call this during inference. Held-out rows are copied unchanged. An
oracle camera is accepted only after a deterministic held-out-feet check.
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from cv.camera_geometry import segment_distances
from cv.gsr import oracle_homography, project
from cv.pitch_lines import NAMES


def clean_row(row):
    people=[p for p in row['people'] if p['pitch'] is not None and p['role'] in ('player','goalkeeper','referee')]
    if len(people)<10:return row,dict(trusted=False,reason='fewer than ten labelled feet')
    # Independent labels, not a self-fit residual: alternate spatially sorted feet.
    order=sorted(people,key=lambda p:(p['box'][0]+p['box'][2])/2)
    training=order[::2];checking=order[1::2]
    cv2.setRNGSeed(11026)
    camera=oracle_homography(training)
    if camera is None:return row,dict(trusted=False,reason='oracle split cannot constrain camera')
    feet=[((p['box'][0]+p['box'][2])/2,p['box'][3]) for p in checking]
    error=np.linalg.norm(project(camera,feet)-np.array([p['pitch'] for p in checking]),axis=1)
    if not np.isfinite(error).all() or np.median(error)>1 or np.percentile(error,90)>2:
        return row,dict(trusted=False,reason='independent feet disagree',median_m=float(np.median(error)))
    camera=oracle_homography(people)
    lines={};changes=[]
    for name,pts in row['lines'].items():
        target=name
        if name in NAMES and len(pts)>=2:
            world=project(camera,[[p['x']*1920,p['y']*1080] for p in pts])
            distances={n:float(np.median(segment_distances(world,n))) for n in NAMES}
            best=min(distances,key=distances.get)
            if distances[name]>2 and distances[best]<.8:
                target=best;changes.append(dict(original=name,corrected=best,original_distance_m=distances[name],corrected_distance_m=distances[best]))
        lines.setdefault(target,[]).extend(pts)
    # Points from separate fragments are not joined into fictitious polylines:
    # leave existing labels alone if a correction collides with another label.
    if len(lines)!=len(row['lines']):
        collisions=[c for c in changes if c['corrected'] in row['lines']]
        for c in collisions:
            lines[c['corrected']]=row['lines'][c['corrected']]
            lines[c['original']]=row['lines'][c['original']]
        changes=[c for c in changes if c not in collisions]
    return {**row,'lines':lines},dict(trusted=True,check_median_m=float(np.median(error)),changes=changes)


def main():
    source=Path('cv/out/training_20261011/pitch_lines/data/rows.json')
    root=Path('cv/out/audit8_20261011/camera_training');root.mkdir(parents=True,exist_ok=True)
    rows=json.loads(source.read_text());diagnostics=[];clean=[]
    for i,row in enumerate(rows['train']):
        result,d=clean_row(row);clean.append(result);diagnostics.append(dict(image=row['image'],clip=row['clip'],**d))
    output={**rows,'train':clean}
    (root/'rows.json').write_text(json.dumps(output))
    report=dict(training_frames=len(clean),trusted_frames=sum(d['trusted'] for d in diagnostics),
        corrected_frames=sum(bool(d.get('changes')) for d in diagnostics),corrected_lines=sum(len(d.get('changes',[])) for d in diagnostics),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),output_sha256=hashlib.sha256((root/'rows.json').read_bytes()).hexdigest(),
        validation_and_test_unchanged=all(output[p]==rows[p] for p in ['val','test']),
        conditions='Supervised training label cleanup only. Foot labels are never camera-inference inputs. Deterministic independent feet check; uncertain rows retain original labels.',diagnostics=diagnostics)
    (root/'label_cleanup.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k!='diagnostics'})


if __name__=='__main__':main()
