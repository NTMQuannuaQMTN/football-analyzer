"""Select image-only appearance association on fixed validation raw detections."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from cv.appearance_tracker import AppearanceBOTSORT
from cv.audit_tracking import evaluate,configurations
from cv.gsr_train import crop,to_tensor
from cv.person_embedding import PersonEmbedding
from cv.session_report import table


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--part',choices=['val','test'],required=True)
    ap.add_argument('--checkpoint',default='cv/out/audit8_20261011/person_embedding/best.pt')
    ap.add_argument('--all-validation',action='store_true',help='Evaluate already frozen choices on all forty validation clips without reselection')
    ap.add_argument('--out',type=Path);a=ap.parse_args()
    original=Path('cv/out/audit8_20261011/appearance_tracking')
    if a.all_validation and (a.part!='val' or a.out is None or a.out.resolve()==original.resolve()):
        raise ValueError('Full validation requires the validation split and a separate explicit --out')
    root=a.out or original;root.mkdir(parents=True,exist_ok=True)
    weights=Path(a.checkpoint);digest=hashlib.sha256(weights.read_bytes()).hexdigest()
    net=PersonEmbedding();net.load_state_dict(torch.load(weights,weights_only=False)['state']);net.eval();torch.set_num_threads(1)
    cache=original/'embeddings'/digest[:16];cache.mkdir(parents=True,exist_ok=True)
    def features_for(image,raw,clip,stem):
        path=cache/f'{clip}_{stem}.npy'
        if path.exists():return np.load(path)
        values=np.zeros((len(raw),128),np.float32);images=[];indices=[]
        for i,b in enumerate(raw):
            patch=crop(image,b[:4])
            if patch is not None:images.append(patch);indices.append(i)
        if images:
            with torch.no_grad():values[indices]=net(to_tensor(np.array(images))).numpy()
        np.save(path,values);return values
    baseline=json.loads(Path('cv/out/audit8_20261011/tracking_generic/selection.json').read_text())['configuration']
    configs={'motion_only':baseline}
    if a.all_validation:
        selection=json.loads((original/'selection.json').read_text())
        if selection['checkpoint_sha256']!=digest:raise ValueError('Expanded validation must use the frozen checkpoint')
        configs['legacy']=configurations()['legacy']
        if selection['configuration'] is not None:configs['selected']=selection['configuration']
    elif a.part=='val':
        for cosine in [.90,.95]:
            for proximity in [.3,.5]:
                configs[f'cos{cosine}_iou{proximity}']={**baseline,'with_reid':True,'model':'auto','minimum_cosine':cosine,'appearance_margin':.05,'proximity_thresh':proximity}
    else:
        selection=json.loads((root/'selection.json').read_text())
        if selection['checkpoint_sha256']!=digest:raise ValueError('Test checkpoint differs from validation-selected checkpoint')
        if selection['configuration'] is not None:configs['selected']=selection['configuration']
    output=evaluate(a.part,'cv/weights/yolo11m.pt',root,configs,1000 if a.all_validation else 4 if a.part=='val' else 18,
        features_for=features_for,tracker_class=AppearanceBOTSORT,raw_cache_root='cv/out/audit8_20261011/tracking_generic')
    output.update(embedding_checkpoint=str(weights),embedding_sha256=digest,appearance_conditions='Current/past detected torso crops only; conservative spatial and mutual-uniqueness gates. No labelled identities, team labels or future crops enter association.')
    (root/f'{a.part}.json').write_text(json.dumps(output,indent=2))
    if a.part=='val' and not a.all_validation:
        base=output['results']['motion_only']
        eligible=[k for k,r in output['results'].items() if k!='motion_only' and r['precision']>=base['precision']-.005 and r['identity_f1']>=base['identity_f1']+.005]
        name=max(eligible,key=lambda k:output['results'][k]['identity_f1']) if eligible else None
        (root/'selection.json').write_text(json.dumps(dict(name=name,configuration=configs[name] if name else None,checkpoint_sha256=digest,
            validation=output['results'][name] if name else None,criterion='Validation identity F1 gain >=0.5 percentage points; precision loss <=0.5 percentage points against selected motion-only tracker'),indent=2))
    if a.all_validation:
        base=output['results']['motion_only'];candidate=output['results'].get('selected')
        gate=bool(candidate and candidate['precision']>=base['precision']-.005 and candidate['identity_f1']>=base['identity_f1']+.005)
        legacy=output['results']['legacy']
        motion_gate=bool(base['precision']>=legacy['precision']-.015 and base['identity_f1']>=legacy['identity_f1'])
        (root/'gate_result.json').write_text(json.dumps(dict(passed=gate,
            criterion=selection['criterion'],scope='Frozen choices; expanded validation only. No reselection, threshold changes or test inspection.',
            motion_vs_legacy_gate=motion_gate,motion_vs_legacy_criterion='Original motion selection requires precision loss <=1.5 percentage points and non-decreasing identity F1',
            summary=output['results']),indent=2))
        rows=[[name,r['gt'],r['pred'],f"{r['precision']:.2%}",f"{r['recall']:.2%}",
               f"{r['identity_f1']:.2%}",r['switches'],r['transitions']] for name,r in output['results'].items()]
        (root/'review.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<title>Expanded anonymous tracking validation</title><style>body{font:16px system-ui;max-width:1200px;'
            'margin:30px auto;padding:16px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #aaa;padding:8px}</style>'
            '<h1>Expanded anonymous tracking validation</h1><p>All forty validation clips, 6,000 frames, '
            'same frozen stock detections and already selected configurations; no new search or selection. '
            'Reference image boxes/IDs score only. No pitch-camera or named-player accuracy is established. '
            f'Appearance-versus-motion gate: {"passed" if gate else "failed"}. '
            f'Original motion-versus-legacy precision gate: {"passed" if motion_gate else "failed"}. '
            'A passing appearance gate does not erase the combined pipeline precision trade-off. No promotion.</p>'+table(
            ['Tracker','GT people','Predictions','Precision','Recall','Anonymous identity F1','ID switches','Matched transitions'],rows)+'</html>')
    print(output['results'],flush=True)


if __name__=='__main__':main()
