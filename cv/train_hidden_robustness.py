"""Isolated clean/noisy-input fine tuning; validation selects, never test.

The corruption is a controlled training augmentation, not an EPL error model.
All unavailable measurements remain NaN; complete coordinates are targets only.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np
import torch

from backend.reconstruction import ml
from backend.reconstruction.datasets import L, W, parse_dfl
from backend.reconstruction.estimator import reconstruct
from backend.reconstruction.real_benchmark import make_mask, segment_view, split


def noisy_measurements(positions, visible, seed):
    rng = np.random.default_rng(seed)
    measured = np.where(visible[..., None], positions, np.nan).copy()
    # Independent foot error plus temporally coherent, anonymous camera error.
    measured[visible] += rng.normal(0, 2/np.sqrt(np.pi/2), (visible.sum(), 2))
    for start in range(0, len(visible), 60):
        measured[start:start+60] += rng.normal(0, 1.5, 2)
    assert not np.isfinite(measured[~visible]).any()
    return measured


def make_rows(job):
    match, half, mask, noisy, seed, stride = job
    info, halves = parse_dfl(match)
    seg = halves[half]
    visible = make_mask(seg, mask, seed)
    on = np.isfinite(seg['pos']).all(2)
    measured = noisy_measurements(seg['pos'], visible, seed+17000) if noisy else np.where(visible[..., None], seg['pos'], np.nan)
    result = {}
    for mode in ['live', 'replay']:
        view, _, roles, anchors = segment_view(info, seg)
        view.xy = measured
        prior = reconstruct(view, visible, mode, {}, on=on, role=roles, anchors=anchors).pos
        x, indices = ml.features(measured, visible, on, seg['team'], seg['role'], seg['ball'], seg['poss'], seg['alive'], prior,
                                 mode, version=4, t_index=range(0, len(visible), stride))
        t, player = indices[:, 0], indices[:, 1]
        correction = seg['pos'][t, player]-prior[t, player]
        correction = np.where((seg['team'][player]=='B')[:, None], -correction, correction)
        result[mode] = dict(X=x, Y=(correction/20).astype(np.float32))
    return job, result


def build(matches, masks, stride, root, label):
    configuration = dict(matches=matches, masks=masks, stride=stride, seed=11026,
        dependency_sources_sha256={name:hashlib.sha256(Path(ml.__file__).with_name(name).read_bytes()).hexdigest()
                                   for name in ('ml.py','real_benchmark.py','datasets.py','estimator.py')},
        sources={name:hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
                 for name,fn in [('noise',noisy_measurements),('rows',make_rows),('features',ml.features),('prior',reconstruct),('masks',make_mask)]})
    digest = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()[:16]
    path = root/f'{label}_{digest}.npz'
    if path.exists():
        return np.load(path, allow_pickle=True)['data'].item()
    jobs = [(mid, half, mask, noisy, 11026+half*101+mi, stride)
            for mid in matches for half in range(len(parse_dfl(mid)[1])) for mi,mask in enumerate(masks) for noisy in [False, True]]
    rows = {(mode,noisy):[] for mode in ['live','replay'] for noisy in [False,True]}
    with ProcessPoolExecutor(max_workers=2) as pool:
        for job, values in pool.map(make_rows, jobs):
            print('features',job, {m:len(v['X']) for m,v in values.items()}, flush=True)
            for mode, value in values.items():
                rows[mode,job[3]].append(value)
    data = {key:{field:np.concatenate([v[field] for v in values]) for field in ['X','Y']} for key,values in rows.items()}
    np.savez_compressed(path, data=np.array(data, dtype=object))
    path.with_suffix('.config.json').write_text(json.dumps(configuration, indent=2))
    return data


def errors(net, data):
    correction, _ = ml.predict(net,data['X'])
    return np.linalg.norm(correction-data['Y']*20,axis=1)


def fine_tune(checkpoint, train, validation, root, mode, variant, deadline, max_epochs=30, early_stopping=True):
    ck = torch.load(checkpoint,weights_only=False)
    net = ml.make_model(ck['n_in']); net.load_state_dict(ck['state'])
    torch.manual_seed(11026)
    opt = torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=2e-4)
    baseline = [float(errors(net,validation[mode,noisy]).mean()) for noisy in [False,True]]
    # Both variants have equal-sized corpora and optimiser update budgets.
    clean = train[mode,False]; noisy = train[mode,True]
    corpus = {key:np.concatenate([clean[key], (noisy if variant=='robust' else clean)[key]]) for key in ['X','Y']}
    x,y = torch.from_numpy(corpus['X']),torch.from_numpy(corpus['Y'])
    history=[]; best_score=np.mean(baseline); best_state=None
    total_updates=0
    for epoch in range(max_epochs):
        if time.time()>=deadline:break
        net.train(); order=torch.randperm(len(x));epoch_updates=0
        for start in range(0,len(x),2048):
            if time.time()>=deadline:break
            b=order[start:start+2048];mu,ls=net(x[b]);ls=ls.clamp(-4,4)
            d2=((mu.detach()-y[b])**2).sum(1)
            loss=torch.nn.functional.huber_loss(mu,y[b],delta=.25)+.01*(2*ls+d2/(2*torch.exp(2*ls))).mean()
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
            epoch_updates+=1;total_updates+=1
        net.eval(); measured=[float(errors(net,validation[mode,n]).mean()) for n in [False,True]]
        score=float(np.mean(measured)); row=dict(epoch=epoch+1,clean_mae_m=measured[0],noisy_mae_m=measured[1],score_m=score)
        row.update(optimizer_updates=epoch_updates,complete_training_epoch=epoch_updates==int(np.ceil(len(x)/2048)))
        history.append(row);print(mode,variant,row,flush=True)
        if measured[0]<=baseline[0]+.3 and score<best_score:
            best_score=score;best_state={k:v.detach().clone() for k,v in net.state_dict().items()}
        if early_stopping and len(history)>=15 and min(r['score_m'] for r in history[:-10])<=min(r['score_m'] for r in history[-10:]):break
    result=dict(mode=mode,variant=variant,baseline_clean_mae_m=baseline[0],baseline_noisy_mae_m=baseline[1],history=history,
                eligible=best_state is not None,maximum_epochs=max_epochs,early_stopping=early_stopping,
                total_optimizer_updates=total_updates,
                selection='Mean clean/noisy validation MAE improves; clean regression <=0.3m; equal maximum clean-control budget (actual epochs may differ with early stopping)')
    if best_state is not None:
        net.load_state_dict(best_state);net.eval()
        vx=np.concatenate([validation[mode,n]['X'] for n in [False,True]])
        vy=np.concatenate([validation[mode,n]['Y'] for n in [False,True]])
        correction,sigma=ml.predict(net,vx);cal=ml.calibrate(sigma,np.linalg.norm(correction-vy*20,axis=1))
        path=root/f'hidden_{mode}_robustness_{variant}_v{ck.get("feat_version",4)}.pt'
        torch.save({**ck,'state':best_state,'calibration':cal,'augmentation':variant,'source_checkpoint':str(checkpoint),'seed':11026},path)
        result.update(checkpoint=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),selected_clean_mae_m=float(errors(net,validation[mode,False]).mean()),selected_noisy_mae_m=float(errors(net,validation[mode,True]).mean()))
    (root/f'{mode}_{variant}.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--deadline',type=float,required=True)
    parser.add_argument('--out',default='cv/out/audit8_20261011/hidden_robustness');args=parser.parse_args()
    root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    train,val,test=split(0)
    selection=json.loads(Path('cv/out/training_20261011/reconstruction/results.json').read_text())
    config=dict(train=train,val=val,test=test,feature_version=4,seed=11026,deadline=args.deadline,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),conditions='Exact identities, known ball/phase. Hidden coordinates targets only. Fixed simulated masks. Test already reviewed for diagnosis; never used for updates, selection or recalibration. No EPL generalisation claim or production promotion.')
    (root/'config.json').write_text(json.dumps(config,indent=2))
    tr=build(train,['sc_camera','k=10','k=6'],5,root,'train')
    va=build([val],['sc_camera','k=10'],3,root,'validation')
    results=[]
    for mode in ['live','replay']:
        checkpoint=Path('cv/out/training_20261011/reconstruction')/selection['selected_test'][mode]['checkpoint']
        for variant in ['clean_control','robust']:
            if time.time()>=args.deadline:break
            results.append(fine_tune(checkpoint,tr,va,root,mode,variant,args.deadline))
            (root/'results.json').write_text(json.dumps(results,indent=2))
    (root/'finished.json').write_text(json.dumps(dict(finished_epoch=time.time(),experiments=len(results))))


if __name__=='__main__':main()
