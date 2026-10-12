"""Matched, within-match temporal-block comparison of fixed-budget candidates.

Bootstrap intervals diagnose correlated evaluation errors within this match;
they do not establish population uncertainty, EPL transfer or causal tactics.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from cv.audit_ball_conditioning import CONDITIONS
from cv.session_report import table


def block_comparison(control,control_indices,candidate,candidate_indices,block_seconds,draws=2000):
    if not np.array_equal(control_indices,candidate_indices):
        raise ValueError('Paired comparison requires identical half/time/player indices in identical order')
    if control.shape!=candidate.shape or len(control)!=len(control_indices):
        raise ValueError('Paired error arrays and indices disagree')
    delta=np.asarray(candidate,float)-control
    if not np.isfinite(delta).all():raise ValueError('Nonfinite paired scores require explicit exclusion before comparison')
    if block_seconds<=0 or draws<=0:raise ValueError('Positive block length and bootstrap draws required')
    rng=np.random.default_rng(11026);sums=np.zeros(draws);counts=np.zeros(draws);blocks=0
    for half in np.unique(control_indices[:,0]):
        selected=control_indices[:,0]==half
        times=control_indices[selected,1];values=delta[selected]
        labels=times//block_seconds;unique=np.unique(labels)
        totals=np.array([values[labels==label].sum() for label in unique])
        sizes=np.array([(labels==label).sum() for label in unique])
        chosen=rng.integers(0,len(unique),(draws,len(unique)))
        sums+=totals[chosen].sum(1);counts+=sizes[chosen].sum(1);blocks+=len(unique)
    estimates=sums/counts
    return dict(n=len(delta),candidate_minus_control_mean_m=float(delta.mean()),block_seconds=block_seconds,
        temporal_blocks=blocks,bootstrap_draws=draws,interval_95_m=np.percentile(estimates,[2.5,97.5]).tolist())


def main():
    root=Path('cv/out/audit8_20261011/context_fixed_budget')
    control_file=root/'evaluation_clean_control/paired_scores.npz'
    candidate_file=root/'evaluation_robust/paired_scores.npz'
    rows=[];results={};training=json.loads((root/'results.json').read_text())
    for mode in ['live','replay']:
        records=[r for r in training if r['mode']==mode]
        if len(records)!=2 or len({r['total_optimizer_updates'] for r in records})!=1:
            raise ValueError('Fixed-budget comparison needs exactly equal recorded optimiser update counts')
        if any(len(r['history'])!=30 or not all(h['complete_training_epoch'] for h in r['history']) for r in records):
            raise ValueError('Fixed-budget comparison requires thirty complete epochs per variant')
    with np.load(control_file) as control,np.load(candidate_file) as candidate:
        for mode in ['live','replay']:
            results[mode]={}
            for condition in CONDITIONS:
                key=f'{mode}/{condition}';c=control[key+'/error_m'];r=candidate[key+'/error_m']
                ci=control[key+'/indices'];ri=candidate[key+'/indices']
                results[mode][condition]={}
                for seconds in [60,120,300]:
                    result=block_comparison(c,ci,r,ri,seconds)
                    results[mode][condition][str(seconds)]=result
                    rows.append([mode,condition,result['n'],seconds,result['temporal_blocks'],
                        round(result['candidate_minus_control_mean_m'],3),
                        f"{result['interval_95_m'][0]:.3f} to {result['interval_95_m'][1]:.3f}"])
    conditions=('Same 97,116 exact half/time/player indices per mode/condition. Negative change means '
        'lower candidate error. Seed11026, 2,000 bootstrap draws, contiguous 60/120/300s temporal '
        'blocks, resampled separately within each half and pooled with scored-person count weights. '
        'One previously reviewed held-out match; overlapping/correlated trajectories and cross-block '
        'dependence may remain. Intervals are approximate within-match diagnostics, not across-match '
        'generalisation intervals, nominal uncertainty regions or evidence of EPL accuracy. Block '
        'lengths are sensitivity checks, not used to tune a model. Both variants have exactly thirty '
        'complete epochs and equal recorded optimiser updates; validation alone selects checkpoints.')
    output=dict(results=results,conditions=conditions,sources_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [control_file,candidate_file,root/'results.json']})
    (root/'paired_comparison.json').write_text(json.dumps(output,indent=2))
    (root/'paired_review.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<title>Matched context training comparison</title><style>body{font:16px system-ui;max-width:1250px;'
        'margin:30px auto;padding:16px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #aaa;padding:8px}</style>'
        '<h1>Matched context training comparison</h1><p>'+conditions+'</p>'+table(
        ['Mode','Context','Hidden samples','Block seconds','Blocks','Candidate minus control mean m',
         'Approximate 95% within-match bootstrap interval m'],rows)+'</html>')
    print(rows,flush=True)


if __name__=='__main__':main()
