"""Local whole-game ReID experiment; hard negatives are teammates.

Uses labelled identities for supervision only. No test or future observations
select the embedding; deployed association consumes past/current crops only.
"""
import argparse
import hashlib
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from cv.gsr import clips,load_clip
from cv.gsr_train import to_tensor
from cv.person_embedding import PersonEmbedding,batch_hard_loss


def metadata():
    result={}
    for split in ['train','valid']:
        for clip in clips(split):
            info,frames=load_clip(clip)
            for f in frames:
                for p in f['people']:
                    if p['role'] not in ('player','goalkeeper') or not str(p['jersey']).isdigit() or not 1<=int(p['jersey'])<=99:continue
                    group=f"{info['game_id']}:{info['game_time_start'].split(' ')[0]}:{p['team']}"
                    result[f"{clip.name}:{p['track']}"]=(group+':'+str(p['jersey']),group)
    return result


def predict(model,images,device):
    result=[];model.eval()
    with torch.no_grad():
        for start in range(0,len(images),128):result.append(model(to_tensor(images[start:start+128]).to(device)).cpu().numpy())
    return np.concatenate(result)


def retrieval(z,tracks,identities):
    """First/last portions of each track; teammate gallery restricts kit shortcuts.

    Evaluation is retrospective retrieval, not a causal tracking score.
    """
    groups=defaultdict(list)
    for i,t in enumerate(tracks):groups[t].append(i)
    gallery=[];queries=[]
    for track,indices in groups.items():
        if len(indices)<6:continue
        n=len(indices)//3
        g=z[indices[:n]].mean(0);g/=max(1e-8,np.linalg.norm(g))
        q=z[indices[-n:]].mean(0);q/=max(1e-8,np.linalg.norm(q))
        identity,team=identities[track];gallery.append((g,identity,team));queries.append((q,identity,team))
    correct=total=0;per_team=defaultdict(lambda:[0,0])
    for q,identity,team in queries:
        candidates=[r for r in gallery if r[2]==team]
        if len({r[1] for r in candidates})<2:continue
        picked=max(candidates,key=lambda r:float(q@r[0]))
        total+=1;correct+=picked[1]==identity;per_team[team][0]+=picked[1]==identity;per_team[team][1]+=1
    return dict(correct=int(correct),queries=total,top1=correct/max(1,total),per_team=dict(per_team),
                conditions='Retrospective first/last track-third retrieval within same match/half/team; not named identity verification or causal tracking accuracy')


def cross_clip_retrieval(z,tracks,identities):
    """Require another clip in the gallery, excluding same-shot shortcuts."""
    groups=defaultdict(list)
    for i,t in enumerate(tracks):groups[t].append(i)
    records=[]
    for track,indices in groups.items():
        if len(indices)<6:continue
        vector=z[indices].mean(0);vector/=max(1e-8,np.linalg.norm(vector))
        identity,team=identities[track]
        records.append((vector,identity,team,track.split(':')[0]))
    correct=total=0;per_team=defaultdict(lambda:[0,0])
    for vector,identity,team,clip in records:
        gallery=[r for r in records if r[2]==team and r[3]!=clip]
        if len({r[1] for r in gallery})<2 or identity not in {r[1] for r in gallery}:continue
        picked=max(gallery,key=lambda r:float(vector@r[0]))
        total+=1;correct+=picked[1]==identity;per_team[team][0]+=picked[1]==identity;per_team[team][1]+=1
    return dict(correct=int(correct),queries=total,top1=correct/max(1,total),per_team=dict(per_team),
                conditions='Leave-clip-out retrospective mean-track retrieval within match/half/team; excludes same-clip gallery. Not causal tracking or named identity verification.')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--hours',type=float,default=1)
    ap.add_argument('--deadline',type=float);ap.add_argument('--device',default='mps')
    ap.add_argument('--out',default='cv/out/audit8_20261011/person_embedding');a=ap.parse_args()
    root=Path(a.out)
    if (root/'config.json').exists():raise ValueError('Use a fresh training output directory')
    root.mkdir(parents=True,exist_ok=True)
    meta_path=root/'metadata.json'
    if not meta_path.exists():meta_path.write_text(json.dumps(metadata()))
    identities=json.loads(meta_path.read_text())
    if a.prepare_only:print('Mapped existing crop tracks',len(identities));return
    data=Path('cv/out/training_20261011/jersey/data')
    with np.load(data/'train.npz') as d:X=d['X'];tracks=d['track']
    with np.load(data/'val.npz') as d:VX=d['X'];vtracks=d['track']
    torch.manual_seed(11026);torch.set_num_threads(1);rng=np.random.default_rng(11026)
    ids=sorted({identities[t][0] for t in tracks});label_of={name:i for i,name in enumerate(ids)}
    indexes=defaultdict(list);by_team=defaultdict(set)
    for i,t in enumerate(tracks):identity,team=identities[t];indexes[identity].append(i);by_team[team].add(identity)
    teams=[t for t,s in by_team.items() if len(s)>=6]
    model=PersonEmbedding().to(a.device);cached=Path('/Users/qnrj/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth')
    if not cached.exists():raise FileNotFoundError('Local weights required; no download attempted')
    weights=torch.load(cached,weights_only=True);weights={k:v for k,v in weights.items() if not k.startswith('fc.')}
    model.encoder.load_state_dict(weights,strict=False)
    head=nn.Linear(128,len(ids)).to(a.device)
    optimizer=torch.optim.AdamW([*model.parameters(),*head.parameters()],lr=.00005,weight_decay=.0001)
    # Fixed validation crop sample spans every track without test information.
    val_indices=[]
    for t in sorted(set(vtracks.tolist())):
        inds=np.flatnonzero(vtracks==t);val_indices.extend(inds[np.linspace(0,len(inds)-1,min(12,len(inds)),dtype=int)].tolist())
    val_indices=np.array(sorted(val_indices));VX=VX[val_indices];vtracks=vtracks[val_indices]
    baseline_z=predict(model,VX,a.device)
    baseline=dict(same_track=retrieval(baseline_z,vtracks,identities),cross_clip=cross_clip_retrieval(baseline_z,vtracks,identities))
    (root/'baseline_validation.json').write_text(json.dumps(baseline,indent=2))
    started=time.time()
    config=dict(seed=11026,device=a.device,hours=a.hours,deadline=a.deadline,started_utc_epoch=started,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),local_pretrained_sha256=hashlib.sha256(cached.read_bytes()).hexdigest(),
        train_games=['4','6','9'],validation_games=['3','5'],test_game='2',selection='Leave-clip-out within-half/team validation top1 retrieval; no test input',
        training_crops=len(X),validation_crops=len(VX),augmentation='Brightness only',test_evaluated=False)
    (root/'config.json').write_text(json.dumps(config,indent=2))
    deadline=min(time.time()+a.hours*3600,a.deadline or float('inf'));history=[];best=-1
    for epoch in range(100):
        if time.time()>=deadline:break
        model.train()
        for layer in model.encoder.modules():
            if isinstance(layer,nn.BatchNorm2d):layer.eval()
        losses=[]
        for step in range(150):
            if time.time()>=deadline:break
            team=teams[int(rng.integers(len(teams)))];chosen=rng.choice(sorted(by_team[team]),6,replace=False)
            selected=np.concatenate([rng.choice(indexes[identity],4,replace=len(indexes[identity])<4) for identity in chosen])
            labels=torch.tensor([label_of[identities[tracks[i]][0]] for i in selected],device=a.device)
            xb=X[selected].astype(np.float32)*rng.uniform(.85,1.15,(len(selected),1,1,1))
            z=model(to_tensor(np.clip(xb,0,255).astype(np.uint8)).to(a.device))
            # Classify within the same half/team only: the same physical player
            # in another half is not falsely made a global negative identity.
            group_ids=sorted(label_of[s] for s in by_team[team])
            target=torch.tensor([group_ids.index(int(label)) for label in labels],device=a.device)
            loss=batch_hard_loss(z,labels)+.2*nn.functional.cross_entropy(head(z)[:,group_ids]*10,target)
            optimizer.zero_grad();loss.backward();optimizer.step();losses.append(loss.item())
        vz=predict(model,VX,a.device)
        result=cross_clip_retrieval(vz,vtracks,identities)
        row=dict(epoch=epoch+1,loss=float(np.mean(losses)) if losses else None,**result,same_track=retrieval(vz,vtracks,identities));history.append(row);print(row,flush=True)
        if result['top1']>best:
            best=result['top1'];torch.save(dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},seed=11026,validation=result,
                train_games=['4','6','9'],validation_games=['3','5'],test_game='2',test_evaluated=False),root/'best.pt')
        (root/'history.json').write_text(json.dumps(history,indent=2))
        if len(history)>20 and np.argmax([r['top1'] for r in history])<len(history)-12:break
    print('Finished; test remains unevaluated',flush=True)
    (root/'finished.json').write_text(json.dumps(dict(started_utc_epoch=started,finished_utc_epoch=time.time(),epochs=len(history),validation_best_top1=best,test_evaluated=False),indent=2))


if __name__=='__main__':main()
