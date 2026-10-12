"""Bounded fresh V3 context-training comparison with disjoint whole-match gates.

Seven existing German tracking matches; no EPL labels or model promotion.
Each fold reserves a separate validation match and excludes its test match.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from backend.reconstruction import ml
from backend.reconstruction.datasets import dfl_matches
from cv.train_context_robustness import build
from cv.train_hidden_robustness import fine_tune


def partitions(matches,index):
    if len(matches)<3 or not 0<=index<len(matches) or len(set(matches))!=len(matches):
        raise ValueError('Unique matches and a valid fold index are required')
    test=matches[index];val=matches[(index+1)%len(matches)]
    return [m for m in matches if m not in [test,val]],val,test


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--deadline',type=float,required=True)
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    if args.epochs<1 or args.deadline<=time.time():raise ValueError('Positive epoch budget and future deadline required')
    if (args.out/'config.json').exists():raise ValueError('Use a fresh output directory')
    torch.set_num_threads(2);args.out.mkdir(parents=True,exist_ok=True)
    cache=args.out/'features';cache.mkdir(exist_ok=True)
    matches=sorted(dfl_matches());started=time.time()
    if len(matches)!=7:raise ValueError('This frozen comparison expects the seven audited matches')
    config=dict(matches=matches,seed=11026,feature_version=3,epochs=args.epochs,deadline_epoch=args.deadline,
        started_epoch=started,training_stride_s=10,validation_stride_s=5,
        validation_gate=f'Both mode candidates: clean validation error <= clean-control +0.3m and mean clean/stale-context validation error improves. {args.epochs} complete epochs and equal optimizer updates required. No held-out candidate evaluation after a failed gate.',
        conditions='Random initialisation, existing V3 architecture, exact visible IDs/roles/phase, simulated masks, no pretrained-match leakage. Each fold uses five training matches, one separate validation match and one excluded test match. Stale ball every30s and unknown possession in alternate60s blocks. Earlier sessions have reviewed these test matches; this is not a fresh external/EPL test. Parameters fixed before any new test output. No deployment or coordinate repair.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.out/'config.json').write_text(json.dumps(config,indent=2))
    (args.out/'training_source.py').write_bytes(Path(__file__).read_bytes())
    training_data={};validation_data={};results=[]
    for index in range(len(matches)):
        if time.time()>=args.deadline:break
        train,val,test=partitions(matches,index);root=args.out/f'fold_{index}';root.mkdir(exist_ok=True)
        for match in train:
            if match not in training_data:
                training_data[match]=build([match],['sc_camera','k=10','k=6'],10,cache,'train_'+match,version=3)
        if val not in validation_data:
            validation_data[val]=build([val],['sc_camera','k=10'],5,cache,'validation_'+val,version=3)
        tr={key:{field:np.concatenate([training_data[match][key][field] for match in train])
                 for field in ['X','Y']} for key in training_data[train[0]]}
        va=validation_data[val];records=[];gates={}
        for mode in ['live','replay']:
            torch.manual_seed(11026+index);net=ml.make_model(tr[mode,False]['X'].shape[1])
            checkpoint=root/f'initial_{mode}.pt'
            torch.save(dict(state=net.state_dict(),n_in=tr[mode,False]['X'].shape[1],feat_version=3,
                calibration={},mode=mode,train=train,val=val,test=test,seed=11026+index),checkpoint)
            for variant in ['clean_control','robust']:
                if time.time()>=args.deadline:break
                record=fine_tune(checkpoint,tr,va,root,mode,variant,args.deadline,max_epochs=args.epochs,early_stopping=False)
                records.append(record)
            current={r['variant']:r for r in records if r['mode']==mode}
            complete=len(current)==2 and all(r['eligible'] and len(r['history'])==args.epochs and all(h['complete_training_epoch'] for h in r['history']) for r in current.values())
            if complete:
                a,b=current['clean_control'],current['robust']
                complete=a['total_optimizer_updates']==b['total_optimizer_updates']
                gates[mode]=bool(complete and b['selected_clean_mae_m']<=a['selected_clean_mae_m']+.3 and
                    b['selected_clean_mae_m']+b['selected_noisy_mae_m']<a['selected_clean_mae_m']+a['selected_noisy_mae_m'])
            else:gates[mode]=False
        result=dict(fold=index,train=train,val=val,test=test,training=records,mode_validation_gates=gates,
            held_out_candidate_status='not run: one or both mode gates failed',evaluations=[])
        (root/'training_results.json').write_text(json.dumps(result,indent=2))
        if len(records)==4 and all(r['eligible'] for r in records):
            variants=['clean_control','robust'] if all(gates.values()) else ['clean_control']
            for variant in variants:
                if args.deadline-time.time()<60:break
                command=[sys.executable,'-m','cv.audit_ball_conditioning','--match',test,
                    '--live-checkpoint',str(root/f'hidden_live_robustness_{variant}_v3.pt'),
                    '--replay-checkpoint',str(root/f'hidden_replay_robustness_{variant}_v3.pt'),
                    '--out',str(root/f'evaluation_{variant}')]
                with (root/f'evaluation_{variant}.log').open('w') as log:
                    try:completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,args.deadline-time.time()))
                    except subprocess.TimeoutExpired:
                        result['evaluations'].append(dict(variant=variant,command=command,status='not completed: deadline'));break
                result['evaluations'].append(dict(variant=variant,command=command,exit_code=completed.returncode))
                if variant=='robust' and completed.returncode==0:
                    result['held_out_candidate_status']='executed after both frozen validation gates passed'
        (root/'training_results.json').write_text(json.dumps(result,indent=2));results.append(result)
        (args.out/'results.json').write_text(json.dumps(results,indent=2));print('fold_finished',index,gates,result['held_out_candidate_status'],flush=True)
    complete=sum(len(r['training'])==4 and all(len(t['history'])==args.epochs and all(h['complete_training_epoch'] for h in t['history']) for t in r['training']) for r in results)
    (args.out/'finished.json').write_text(json.dumps(dict(finished_epoch=time.time(),attempted_folds=len(results),complete_training_folds=complete,deadline_epoch=args.deadline)))


if __name__=='__main__':main()
