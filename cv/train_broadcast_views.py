"""Private bounded training of wide-sideline appearance versus unsupported views.

Frozen locally cached ImageNet encoder + learned logistic head. This model
does not calibrate coordinates, recognise a named player or prove live time.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18

from cv.broadcast_views import samples
from cv.session_report import table
from cv.soccernet import DEFAULT_GAME,camera_segments,game_dir,labels


def metrics(prob,target,threshold):
    pred=prob>=threshold;positive=target==1
    tp=int((pred&positive).sum());fp=int((pred&~positive).sum())
    return dict(n=len(target),sideline_n=int(positive.sum()),unsupported_n=int((~positive).sum()),
        accepted_n=int(pred.sum()),true_sideline_n=tp,false_accepts_n=fp,
        precision=tp/max(1,tp+fp),recall=tp/max(1,int(positive.sum())),
        false_accept_rate=fp/max(1,int((~positive).sum())),accuracy=float((pred==positive).mean()))


def choose_threshold(prob,target):
    rows=[dict(threshold=t,**metrics(prob,target,t)) for t in (.5,.7,.85,.95,.99)]
    eligible=[r for r in rows if r['accepted_n']>=20 and r['precision']>=.98 and r['false_accept_rate']<=.05]
    return max(eligible,key=lambda r:r['accepted_n']) if eligible else None,rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--device',default='cpu');ap.add_argument('--epochs',type=int,default=300)
    ap.add_argument('--deadline',type=float,required=True);a=ap.parse_args()
    if not np.isfinite(a.deadline) or a.deadline<=time.time() or a.epochs<1:raise ValueError('Future finite deadline and positive epochs required')
    if a.out.exists():raise ValueError('Use a fresh private output directory')
    a.out.mkdir(parents=True);torch.manual_seed(12026);torch.set_num_threads(2);cv2.setNumThreads(1)
    pretrained=Path(torch.hub.get_dir())/'checkpoints/resnet18-f37072fd.pth'
    if not pretrained.is_file():raise FileNotFoundError('Local encoder missing; no download attempted')
    review=json.loads(Path('cv/out/audit8_20261011/manual_review/frames.json').read_text())
    rows=samples(camera_segments(DEFAULT_GAME,1),[r['video_s'] for r in review])
    event_times=[r['t'] for r in labels(DEFAULT_GAME,1)[1] if r['label'] in ('Penalty','Corner','Direct free-kick','Indirect free-kick')]
    for r in rows:r['near_set_piece']=any(abs(r['video_s']-t)<=15 for t in event_times)
    split_counts={p:dict(n=sum(r['part']==p for r in rows),sideline=sum(r['part']==p and r['target']==1 for r in rows)) for p in ('train','val','test','review')}
    for p in ('train','val'):
        if not 0<split_counts[p]['sideline']<split_counts[p]['n']:raise ValueError('Both view classes required')
    video=game_dir(DEFAULT_GAME)/'1_720p.mkv';annotations=game_dir(DEFAULT_GAME)/'Labels-cameras.json'
    config=dict(seed=12026,device=a.device,deadline_epoch=a.deadline,max_epochs=a.epochs,game=DEFAULT_GAME,
        encoder_sha256=hashlib.sha256(pretrained.read_bytes()).hexdigest(),annotation_sha256=hashlib.sha256(annotations.read_bytes()).hexdigest(),
        video=str(video),video_size=video.stat().st_size,video_mtime_ns=video.stat().st_mtime_ns,
        manual_frames_sha256=hashlib.sha256(Path('cv/out/audit8_20261011/manual_review/frames.json').read_bytes()).hexdigest(),
        split_counts=split_counts,sample_stride_s=3,shot_boundary_margin_s=1,temporal_boundary_buffer_s=15,
        split='Whole shots assigned by midpoint 300s block: validation2/6, test4/8, remaining train; all manual review shots separately withheld',
        replay='All replay-labelled images excluded; repeated replay-origin footage cannot contaminate training/test',
        inputs='Current image only; no ball, player, manual world coordinate, event or future frame input',
        targets='Annotated sideline view category, not good calibration; all other angles/close-ups negative',
        gate='Validation precision>=98%, false-accept rate<=5%, at least20 accepted; widest eligible fixed threshold',
        limitations='One existing EPL match only, temporal rather than match-level holdout. Annotation noise possible. Probabilities uncalibrated. No replay-time detection or global coordinate guarantee.',
        sources_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['cv/train_broadcast_views.py','cv/broadcast_views.py','cv/soccernet.py']})
    (a.out/'config.json').write_text(json.dumps(config,indent=2));(a.out/'samples.json').write_text(json.dumps(rows,indent=2))
    (a.out/'training_source.py').write_bytes(Path(__file__).read_bytes())
    encoder=resnet18(weights=None);encoder.load_state_dict(torch.load(pretrained,weights_only=True));encoder.fc=nn.Identity();encoder.to(a.device).eval()
    cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or fps<=0:raise ValueError('Private video unavailable')
    targets={int(round(r['video_s']*fps)):i for i,r in enumerate(rows)};features=np.empty((len(rows),512),np.float32);grass=np.zeros(len(rows));seen=set();batch=[];indices=[]
    mean=torch.tensor([.485,.456,.406])[None,:,None,None];std=torch.tensor([.229,.224,.225])[None,:,None,None]
    def flush():
        if not batch:return
        tensor=torch.from_numpy(np.stack(batch).transpose(0,3,1,2).copy()).float()/255
        with torch.no_grad():value=encoder(((tensor-mean)/std).to(a.device)).cpu().numpy()
        features[indices]=value;seen.update(indices);batch.clear();indices.clear()
        print('features',len(seen),len(rows),flush=True)
    index=0
    while cap.grab():
        if time.time()>=a.deadline:raise RuntimeError('Deadline during feature extraction; no result claimed')
        if index in targets:
            ok,image=cap.retrieve()
            if not ok:raise RuntimeError('Video read failure')
            i=targets[index];small=cv2.resize(image,(224,126));canvas=np.zeros((224,224,3),np.uint8);canvas[49:175]=cv2.cvtColor(small,cv2.COLOR_BGR2RGB)
            hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV);grass[i]=float(((hsv[:,:,0]>=25)&(hsv[:,:,0]<=95)&(hsv[:,:,1]>35)&(hsv[:,:,2]>30)).mean())
            batch.append(canvas);indices.append(i)
            if len(batch)>=32:flush()
        index+=1
        if index>max(targets):break
    cap.release();flush()
    if len(seen)!=len(rows):raise RuntimeError('Incomplete image extraction')
    np.savez_compressed(a.out/'features.npz',features=features,grass_fraction=grass)
    train=np.array([i for i,r in enumerate(rows) if r['part']=='train']);val=np.array([i for i,r in enumerate(rows) if r['part']=='val'])
    center=features[train].mean(0);scale=features[train].std(0).clip(.01)
    x=torch.from_numpy((features-center)/scale);y=torch.tensor([r['target'] for r in rows],dtype=torch.float32)
    head=nn.Linear(512,1);optimizer=torch.optim.AdamW(head.parameters(),lr=.01,weight_decay=.01)
    history=[];best_key=None;best_state=None;best_epoch=None;best_gate=None
    positive_weight=torch.tensor(float((y[train]==0).sum()/max(1,int((y[train]==1).sum()))))
    for epoch in range(1,a.epochs+1):
        if time.time()>=a.deadline:break
        head.train();logits=head(x[train])[:,0];loss=nn.functional.binary_cross_entropy_with_logits(logits,y[train],pos_weight=positive_weight)
        optimizer.zero_grad();loss.backward();optimizer.step();head.eval()
        with torch.no_grad():logits=head(x[val])[:,0];vl=nn.functional.binary_cross_entropy_with_logits(logits,y[val]).item();prob=logits.sigmoid().numpy()
        selected,thresholds=choose_threshold(prob,y[val].numpy());key=(1,selected['accepted_n'],selected['precision'],-vl) if selected else (0,0,0,-vl)
        record=dict(epoch=epoch,train_loss=loss.item(),validation_loss=vl,selected_threshold=selected,thresholds=thresholds);history.append(record)
        if best_key is None or key>best_key:best_key=key;best_state={k:v.detach().clone() for k,v in head.state_dict().items()};best_epoch=epoch;best_gate=selected
    if best_state is None:raise RuntimeError('No completed training update')
    head.load_state_dict(best_state)
    torch.save(dict(state=best_state,center=center,scale=scale,encoder_path=str(pretrained),encoder_sha256=config['encoder_sha256'],config=config,selected_epoch=best_epoch,threshold=best_gate['threshold'] if best_gate else None),a.out/'best.pt')
    (a.out/'history.json').write_text(json.dumps(history,indent=2))
    with torch.no_grad():prob=head(x)[:,0].sigmoid().numpy()
    parts={}
    for part in ('val','test','review'):
        ids=np.array([i for i,r in enumerate(rows) if r['part']==part])
        if part=='test' and best_gate is None:parts[part]=dict(status='not run: validation gate failed');continue
        if not len(ids):parts[part]=dict(status='no samples');continue
        threshold=best_gate['threshold'] if best_gate else .5
        groups={'all':ids,'near_set_piece':np.array([i for i in ids if rows[i]['near_set_piece']]),'closeups':np.array([i for i in ids if rows[i]['kind']=='closeup']),'alternate_angles':np.array([i for i in ids if rows[i]['kind']=='alternate'])}
        parts[part]=dict(status='scored',threshold=threshold,groups={name:metrics(prob[ii],y[ii].numpy(),threshold) for name,ii in groups.items() if len(ii)},grass_baseline=metrics(grass[ids],y[ids].numpy(),.35),
            predictions=[dict(**rows[i],probability=float(prob[i]),accepted=bool(prob[i]>=threshold)) for i in ids])
    results=dict(config=config,completed_epochs=len(history),best_epoch=best_epoch,validation_gate=best_gate,parts=parts,finished_epoch=time.time(),promoted=False)
    (a.out/'results.json').write_text(json.dumps(results,indent=2))
    display=[]
    for part,value in parts.items():
        for group,r in value.get('groups',{}).items():display.append([part,group,r['n'],r['sideline_n'],r['unsupported_n'],r['accepted_n'],r['false_accepts_n'],round(r['precision'],3),round(r['recall'],3)])
    (a.out/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Close-up view training</title><style>body{font:16px system-ui;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:8px}</style><h1>Close-up / alternate-angle view training</h1><p>Image view-category recognition only, not coordinate accuracy. One EPL match; whole-shot temporal holdout; manual shots withheld. Replays excluded. Annotation targets and probabilities are not verified geometry/confidence.</p>'+table(['Split','Group','N','Sideline N','Unsupported N','Accepted','False accepts','Precision','Recall'],display)+f'<p>Completed epochs: {len(history)}. Validation gate: {"passed" if best_gate else "failed; test not run"}. No deployment.</p>')
    print({k:v for k,v in results.items() if k not in ('parts','config')},flush=True)


if __name__=='__main__':main()
