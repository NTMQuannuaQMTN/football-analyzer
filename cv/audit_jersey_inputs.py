"""Private input/readability diagnostics; no new training or held-out selection."""
import hashlib
from html import escape
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from cv import gsr_train as g
from cv.session_report import table


def main():
    root=Path('cv/out/audit8_20261011/jersey_inputs');root.mkdir(parents=True,exist_ok=True)
    source=Path('cv/out/training_20261011/jersey');parts={};class_rows=[]
    training=np.load(source/'data/train.npz');classes,counts=np.unique(training['y'],return_counts=True)
    majority=int(classes[counts.argmax()]);support=set(map(int,classes))
    for part in ['train','val','test']:
        with np.load(source/f'data/{part}.npz') as dataset:
            labels={}
            for track,number in zip(dataset['track'],dataset['y']):labels.setdefault(str(track),set()).add(int(number))
            unseen=set(map(int,np.unique(dataset['y'])))-support
            value=dict(crops=len(dataset['y']),tracks=len(labels),conflicting_track_labels=sum(len(x)>1 for x in labels.values()),
                classes=len(np.unique(dataset['y'])),numbers_not_in_training=sorted(unseen),
                crops_with_unseen_number=int(np.isin(dataset['y'],list(unseen)).sum()),
                training_majority_number=majority,constant_training_majority_crop_accuracy=float((dataset['y']==majority).mean()),
                constant_training_majority_correct_tracks=sum(x=={majority} for x in labels.values()),
                dataset_sha256=hashlib.sha256((source/f'data/{part}.npz').read_bytes()).hexdigest())
            parts[part]=value
            class_rows.append([part,value['crops'],value['tracks'],value['classes'],value['conflicting_track_labels'],
                ', '.join(map(str,value['numbers_not_in_training'])) or 'None',value['crops_with_unseen_number'],
                f"{value['constant_training_majority_crop_accuracy']:.2%}"])
    if not (Path(torch.hub.get_dir())/'checkpoints/resnet18-f37072fd.pth').is_file():
        raise FileNotFoundError('Local pretrained trunk missing; do not download for this audit')
    torch.set_num_threads(1);model=g.DigitNet.build();model.load_state_dict(torch.load(source/'best.pt',weights_only=False));model.eval()
    with np.load(source/'data/val.npz') as validation:
        indices=np.unique(np.linspace(0,len(validation['y'])-1,1280,dtype=int))
        images=validation['X'][indices];labels=validation['y'][indices];tracks=validation['track'][indices]
        probabilities=g.predict_jersey(model,images,'cpu',bs=128)
    predicted=probabilities.argmax(1);scores=probabilities.max(1);correct=predicted==labels
    selected=[];used=set()
    for kind,mask,reverse,limit in [('Incorrect high score',~correct,True,20),('Correct high score',correct,True,10),('Low score',np.ones(len(indices),bool),False,10)]:
        ordered=sorted(np.flatnonzero(mask),key=lambda i:float(scores[i]),reverse=reverse);count=0
        for i in ordered:
            key=str(tracks[i])
            if key in used:continue
            used.add(key);filename=f'crop_{len(selected)+1:02d}.png'
            cv2.imwrite(str(root/filename),cv2.resize(images[i],(128,192),interpolation=cv2.INTER_NEAREST))
            selected.append(dict(kind=kind,index=int(indices[i]),track=key,reference_number=int(labels[i]),
                predicted_number=int(predicted[i]),uncalibrated_model_score=float(scores[i]),image=filename))
            count+=1
            if count==limit:break
    conditions=('Original four-epoch model and existing whole-game crop splits. No new training, threshold '
        'selection or held-out predictions. Dataset inventory includes test labels only for support/count diagnostics. '
        'Uniform 1,280-crop validation sample; selected gallery is symptom-biased and not an accuracy estimate. '
        'Reference jersey numbers are dataset identity annotations, not per-frame legibility labels. '
        'Front views, blur, occlusion and tiny original torsos may contain no readable number. '
        'Scores are uncalibrated model outputs. Temporal voting is retrospective, not live OCR. Keep GSR images private.')
    old=json.loads((source/'results.json').read_text())
    threshold=old['selected_threshold'];eligible=any(v['track_precision']>=.9 for v in old['validation'].values())
    result=dict(parts=parts,validation_sample_n=len(indices),validation_sample_accuracy=float(correct.mean()),
        gallery=selected,conditions=conditions,original_model_sha256=hashlib.sha256((source/'best.pt').read_bytes()).hexdigest(),
        old_threshold=threshold,old_threshold_met_validation_90pct_precision=eligible,
        old_threshold_note='Original .95 fallback did not meet its .90 validation precision target; it was never promoted')
    (root/'results.json').write_text(json.dumps(result,indent=2))
    cards=''.join('<figure><img src="'+r['image']+'" alt="Private labelled torso crop"><figcaption>'+escape(
        f"{r['kind']} · {r['track']} · dataset #{r['reference_number']} / prediction #{r['predicted_number']} · uncalibrated score {r['uncalibrated_model_score']:.3f}")+'</figcaption></figure>' for r in selected)
    (root/'review.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Private jersey input audit</title>'
        '<style>body{font:16px system-ui;max-width:1250px;margin:30px auto;padding:16px}table{border-collapse:collapse;width:100%}'
        'td,th{border:1px solid #aaa;padding:8px}.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:20px}'
        'figure{margin:0}figcaption{font-size:13px;line-height:1.5}</style><h1>Private jersey input review</h1><p>'+escape(conditions)
        +'</p><p>The original fallback threshold did not reach its validation precision target. No named recognition claim is justified.</p>'
        +table(['Split','Crops','Tracklets','Number classes','Conflicting labels','Unseen training numbers','Unseen-number crops',
            'Constant training-majority crop accuracy'],class_rows)+'<h2>Inspect actual information in the crops</h2><div class="gallery">'+cards+'</div></html>')
    print(root/'review.html',len(selected),'private crops',flush=True)


if __name__=='__main__':main()
