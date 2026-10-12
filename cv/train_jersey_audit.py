"""Bounded local continuation of the existing jersey model, with a real gate."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from cv import gsr_train as g


def select_threshold(validation,min_decisions=20):
    eligible=[(float(threshold),value) for threshold,value in validation.items()
              if value['decided_tracks']>=min_decisions and value['correct_tracks']/value['decided_tracks']>=.9]
    return max(eligible,key=lambda item:item[1]['decided_tracks'])[0] if eligible else None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--hours',type=float,default=1.1)
    parser.add_argument('--deadline',type=float,required=True)
    parser.add_argument('--device',default='mps')
    parser.add_argument('--out',type=Path,default=Path('cv/out/audit8_20261011/jersey_continuation'))
    args=parser.parse_args()
    if args.hours<=0 or args.deadline<=time.time():raise ValueError('Training requires a future bounded deadline')
    source=Path('cv/out/training_20261011/jersey')
    pretrained=Path(torch.hub.get_dir())/'checkpoints/resnet18-f37072fd.pth'
    if not pretrained.is_file():raise FileNotFoundError('Local pretrained trunk missing; no automatic download permitted')
    args.out.mkdir(parents=True,exist_ok=True);g.JERSEY_DIR=source/'data'
    seed=11026;torch.manual_seed(seed);torch.set_num_threads(2);rng=np.random.default_rng(seed)
    with np.load(source/'data/train.npz') as data:images=data['X'];labels=data['y']
    with np.load(source/'data/val.npz') as data:val_images=data['X'];val_labels=data['y'];val_tracks=data['track']
    model=g.DigitNet.build().to(args.device)
    model.load_state_dict(torch.load(source/'best.pt',weights_only=False))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.0005)
    started=time.time();end=min(started+args.hours*3600,args.deadline)
    configuration=dict(seed=seed,device=args.device,started_epoch=started,deadline_epoch=end,
        train_games=['4','6','9'],validation_games=['3','5'],test_games=['2'],
        training_crops=len(labels),validation_crops=len(val_labels),learning_rate=.0001,
        resume_checkpoint=str(source/'best.pt'),resume_sha256=hashlib.sha256((source/'best.pt').read_bytes()).hexdigest(),
        local_pretrained_sha256=hashlib.sha256(pretrained.read_bytes()).hexdigest(),
        inputs_sha256={part:hashlib.sha256((source/f'data/{part}.npz').read_bytes()).hexdigest() for part in ['train','val']},
        gate='Validation track precision >=90% with at least20 decided tracks; choose widest coverage among eligible fixed thresholds .5/.7/.85/.95. No fallback if none eligible.',
        selection='Eligible decision count first; otherwise correct-track count at threshold .5; validation only.',
        limitations='No per-crop legibility labels; aggregate voting is retrospective. Original test game has been examined in earlier sessions; remains excluded from training and selection. Optimizer reset, existing architecture and augmentation.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.out/'config.json').write_text(json.dumps(configuration,indent=2));(args.out/'training_source.py').write_bytes(Path(__file__).read_bytes())
    history=[];best=None;best_epoch=0
    targets=[torch.from_numpy(value.astype(np.int64)) for value in [labels>=10,labels//10,labels%10]]
    ce=torch.nn.functional.cross_entropy
    for epoch in range(1,61):
        if time.time()>=end:break
        model.train();losses=[];seen=0
        for offset in range(0,len(images),128):
            if time.time()>=end:break
            if offset==0:order=rng.permutation(len(images))
            indices=order[offset:offset+128];batch=torch.from_numpy(indices)
            heads=model(g.to_tensor(g.augment(images[indices],rng)).to(args.device))
            truth=[target[batch].to(args.device) for target in targets];two=truth[0]==1
            loss=ce(heads[0],truth[0])+ce(heads[2],truth[2])
            if two.any():loss+=ce(heads[1][two],truth[1][two])
            optimizer.zero_grad();loss.backward();optimizer.step();losses.append(loss.item());seen+=len(indices)
        if not losses:break
        model.eval();probabilities=g.predict_jersey(model,val_images,args.device,bs=256)
        validation={str(threshold):g.jersey_metrics(probabilities,val_labels,val_tracks,threshold) for threshold in [.5,.7,.85,.95]}
        threshold=select_threshold(validation)
        score=(1,validation[str(threshold)]['decided_tracks'],validation[str(threshold)]['correct_tracks']) if threshold is not None else (0,validation['0.5']['correct_tracks'],validation['0.5']['decided_tracks'])
        record=dict(epoch=epoch,training_loss=float(np.mean(losses)),training_samples_seen=seen,complete_training_epoch=seen==len(images),
                    validation=validation,eligible_threshold=threshold,finished_epoch=time.time())
        history.append(record)
        if best is None or score>best:
            best=score;best_epoch=epoch
            torch.save({key:value.detach().cpu().clone() for key,value in model.state_dict().items()},args.out/'best.pt')
        (args.out/'history.json').write_text(json.dumps(history,indent=2));print(record,flush=True)
        if epoch-best_epoch>=8:break
    if not history:raise RuntimeError('Deadline prevented any training batch')
    chosen=history[best_epoch-1];threshold=chosen['eligible_threshold']
    result=dict(config=configuration,completed_epochs=len(history),best_epoch=best_epoch,selected_threshold=threshold,
                validation=chosen['validation'],test=None,test_status='not run: no eligible validation threshold',
                training_finished_epoch=time.time(),checkpoint_sha256=hashlib.sha256((args.out/'best.pt').read_bytes()).hexdigest())
    if threshold is not None and time.time()<args.deadline:
        model.load_state_dict(torch.load(args.out/'best.pt',weights_only=False));model.eval()
        result['test']=g.eval_jersey(model,'test',args.device,conf=threshold)
        result['test_status']='executed once with frozen validation-selected model and threshold'
    elif threshold is not None:result['test_status']='not run: evaluation deadline reached'
    (args.out/'results.json').write_text(json.dumps(result,indent=2));print(result,flush=True)


if __name__=='__main__':main()
