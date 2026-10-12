"""Bounded private V4 continuation with stale ball / unknown possession inputs.

Equal-sized clean-control and context-augmented corpora; fixed whole-match
training/validation split. Test is not read in feature generation or selection.
No deployed checkpoint is changed. This cannot repair camera or named identity.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from backend.reconstruction import ml
from backend.reconstruction.datasets import parse_dfl, reconstruction_input_fingerprints
from backend.reconstruction.estimator import reconstruct
from backend.reconstruction.real_benchmark import make_mask, segment_view, split
from cv.audit_ball_conditioning import context
from cv.train_hidden_robustness import fine_tune


def training_context(ball,possession):
    ball,possession=context(ball,possession,'ball_observed_every_30s')
    possession[(np.arange(len(possession))//60)%2==1]=-1
    return ball,possession


def rows(job):
    match,half,mask,corrupted,seed,stride=job[:6]
    version=job[6] if len(job)>6 else 4
    info,halves=parse_dfl(match);seg=halves[half]
    visible=make_mask(seg,mask,seed);on=np.isfinite(seg['pos']).all(2)
    measured=np.where(visible[...,None],seg['pos'],np.nan)
    ball,possession=training_context(seg['ball'],seg['poss']) if corrupted else (seg['ball'],seg['poss'])
    result={}
    for mode in ['live','replay']:
        view,_,roles,anchors=segment_view(info,seg)
        view.xy=measured;view.ball=ball;view.poss=possession
        prior=reconstruct(view,visible,mode,{},on=on,role=roles,anchors=anchors).pos
        x,idx=ml.features(measured,visible,on,seg['team'],seg['role'],ball,possession,
            seg['alive'],prior,mode,version=version,t_index=range(0,len(visible),stride))
        t,j=idx[:,0],idx[:,1];correction=seg['pos'][t,j]-prior[t,j]
        correction=np.where((seg['team'][j]=='B')[:,None],-correction,correction)
        result[mode]=dict(X=x,Y=(correction/20).astype(np.float32))
    return job,result


def build(matches,masks,stride,root,label,version=4):
    sources={name:hashlib.sha256(Path(path).read_bytes()).hexdigest() for name,path in [
        ('training',__file__),('context','cv/audit_ball_conditioning.py'),
        ('optimization','cv/train_hidden_robustness.py'),('features',ml.__file__),
        ('datasets','backend/reconstruction/datasets.py'),('prior','backend/reconstruction/estimator.py'),
        ('visibility','backend/reconstruction/real_benchmark.py')]}
    config=dict(matches=matches,masks=masks,stride=stride,seed=11026,feature_version=version,sources_sha256=sources,
        consumed_input_sha256=reconstruction_input_fingerprints(matches,masks))
    digest=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
    path=root/f'{label}_{digest}.npz'
    if path.exists():return np.load(path,allow_pickle=True)['data'].item()
    jobs=[(match,half,mask,corrupted,11026+half*101+mi,stride,version)
          for match in matches for half in range(len(parse_dfl(match)[1]))
          for mi,mask in enumerate(masks) for corrupted in [False,True]]
    collected={(mode,corrupted):[] for mode in ['live','replay'] for corrupted in [False,True]}
    with ProcessPoolExecutor(max_workers=2) as pool:
        for job,data in pool.map(rows,jobs):
            for mode,value in data.items():collected[mode,job[3]].append(value)
            print('features',job,{mode:len(v['X']) for mode,v in data.items()},flush=True)
    data={key:{field:np.concatenate([v[field] for v in values]) for field in ['X','Y']}
          for key,values in collected.items()}
    np.savez_compressed(path,data=np.array(data,dtype=object))
    path.with_suffix('.config.json').write_text(json.dumps(config,indent=2))
    return data


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--deadline',type=float,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--fixed-epochs',type=int,help='Equal complete epoch budget, disable validation early stopping')
    args=parser.parse_args()
    if args.fixed_epochs is not None and args.fixed_epochs<=0:raise ValueError('Positive epoch budget required')
    torch.set_num_threads(2);args.out.mkdir(parents=True,exist_ok=True)
    started=time.time();train,val,test=split(0)
    selection=json.loads(Path('cv/out/training_20261011/reconstruction/results.json').read_text())
    config=dict(train=train,val=val,test=test,started_epoch=started,deadline_epoch=args.deadline,
        seed=11026,feature_version=4,train_stride_s=10,validation_stride_s=5,
        fixed_epochs=args.fixed_epochs,
        selection='Existing frozen mean clean/corrupted validation metre error improves; clean regression <=0.3m. Equal-size clean-control update budget; no test tuning.',
        augmentation='Ball observed every30s and held causally; possession unknown during alternate60s blocks. No new missingness feature: existing V4 architecture only.',
        conditions='Whole-match split. Simulated visibility, exact visible positions/IDs/roles/phase. Hidden positions are training targets only. Test has been previously reviewed for diagnostics, excluded from feature building, updates and selection. No EPL generalisation claim or production promotion.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.out/'config.json').write_text(json.dumps(config,indent=2))
    (args.out/'training_source.py').write_bytes(Path(__file__).read_bytes())
    tr=build(train,['sc_camera','k=10','k=6'],10,args.out,'train')
    va=build([val],['sc_camera','k=10'],5,args.out,'validation')
    results=[]
    for mode in ['live','replay']:
        checkpoint=Path('cv/out/training_20261011/reconstruction')/selection['selected_test'][mode]['checkpoint']
        metadata=torch.load(checkpoint,weights_only=False)
        if test in metadata.get('train',[]) or metadata.get('test')!=test:
            raise ValueError('Starting checkpoint does not exclude the evaluation match')
        for variant in ['clean_control','robust']:
            if time.time()>=args.deadline:break
            r=fine_tune(checkpoint,tr,va,args.out,mode,variant,args.deadline,
                max_epochs=args.fixed_epochs or 30,early_stopping=args.fixed_epochs is None)
            r.update(context_profile=config['augmentation'],training_rows=len(tr[mode,False]['X']),
                validation_rows=len(va[mode,False]['X']))
            results.append(r);(args.out/'results.json').write_text(json.dumps(results,indent=2))
    (args.out/'finished.json').write_text(json.dumps(dict(finished_epoch=time.time(),experiments=len(results))))


if __name__=='__main__':main()
