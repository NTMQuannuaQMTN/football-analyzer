"""Controlled hidden-model context stress, not EPL ball or position accuracy."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from backend.reconstruction import ml
from backend.reconstruction.datasets import L, W, parse_dfl
from backend.reconstruction.estimator import reconstruct
from backend.reconstruction.inference import load_ml
from backend.reconstruction.real_benchmark import make_mask, segment_view
from backend.reconstruction.session_evaluation import summarize
from cv.audit_hidden_sensitivity import require_held_out
from cv.session_report import table

ROOT = Path('cv/out/audit8_20261011')
CONDITIONS = ['exact_context', 'ball_observed_every_30s', 'possession_unknown',
              'midfield_ball_and_possession_unknown']


def context(ball, possession, condition):
    ball = np.asarray(ball).copy()
    possession = np.asarray(possession).copy()
    if condition == 'ball_observed_every_30s':
        ball = ball[(np.arange(len(ball)) // 30) * 30]
    elif condition == 'possession_unknown':
        possession[:] = -1
    elif condition == 'midfield_ball_and_possession_unknown':
        ball[:] = [L/2, W/2]
        possession[:] = -1
    elif condition != 'exact_context':
        raise ValueError(condition)
    return ball, possession


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--live-checkpoint',type=Path)
    parser.add_argument('--replay-checkpoint',type=Path)
    parser.add_argument('--out',type=Path,default=ROOT/'ball_conditioning')
    parser.add_argument('--match',help='Explicit excluded evaluation match; requires selected checkpoint paths')
    parser.add_argument('--save-paired-scores',action='store_true',help='Evaluation-only exact indices/errors for paired temporal-block uncertainty checks')
    args=parser.parse_args()
    if bool(args.live_checkpoint)!=bool(args.replay_checkpoint):
        raise ValueError('Supply both validation-selected mode checkpoints')
    if args.match and not args.live_checkpoint:
        raise ValueError('Match override requires explicit validation-selected checkpoints')
    if args.live_checkpoint and args.out.resolve()==(ROOT/'ball_conditioning').resolve():
        raise ValueError('Explicit candidate checkpoints require separate --out; preserve the diagnostic control')
    torch.set_num_threads(1)
    output = args.out
    output.mkdir(parents=True, exist_ok=True)
    match = args.match or 'DFL-MAT-J03WMX'
    info, segments = parse_dfl(match)
    results = {}; display = [];score_arrays={}
    for mode in ['live','replay']:
        checkpoint = getattr(args,f'{mode}_checkpoint') or ROOT/'hidden_robustness_causal_masks'/f'hidden_{mode}_robustness_clean_control_v4.pt'
        require_held_out(torch.load(checkpoint,weights_only=False),match)
        net, calibration, version = load_ml(mode,checkpoint.stem.removeprefix(f'hidden_{mode}_'),checkpoint.parent)
        accum = {condition:dict(error=[],hit=[],ball_error=[],age=[],indices=[]) for condition in CONDITIONS}
        for segment_index, segment in enumerate(segments):
            visible = make_mask(segment,'sc_camera',777+segment_index)
            on = np.isfinite(segment['pos']).all(2)
            measurements = np.where(visible[...,None],segment['pos'],np.nan)
            for condition in CONDITIONS:
                ball, possession = context(segment['ball'],segment['poss'],condition)
                view,_,roles,anchors = segment_view(info,segment)
                view.xy=measurements;view.ball=ball;view.poss=possession
                prior=reconstruct(view,visible,mode,{},on=on,role=roles,anchors=anchors).pos
                features,indices=ml.features(measurements,visible,on,segment['team'],segment['role'],
                    ball,possession,segment['alive'],prior,mode,version=version)
                correction,sigma=ml.predict(net,features);t,j=indices[:,0],indices[:,1]
                predicted=np.clip(prior[t,j]+np.where((segment['team'][j]=='B')[:,None],-correction,correction),[0,0],[L,W])
                error=np.linalg.norm(predicted-segment['pos'][t,j],axis=1)
                radius,_=ml.apply_calibration(calibration,sigma)
                accum[condition]['error'].append(error);accum[condition]['hit'].append(error<=radius)
                accum[condition]['ball_error'].append(np.linalg.norm(ball-segment['ball'],axis=1))
                accum[condition]['age'].append(ml.since_until(visible)[0][t,j])
                accum[condition]['indices'].append(np.column_stack([np.full(len(t),segment_index),t,j]))
                print(mode,segment_index,condition,len(error),flush=True)
        results[mode]=dict(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),conditions={})
        base=np.concatenate(accum['exact_context']['error'])
        base_indices=np.concatenate(accum['exact_context']['indices'])
        for condition,record in accum.items():
            error=np.concatenate(record['error']);age=np.concatenate(record['age'])
            if len(error)!=len(base):raise ValueError('Context comparison requires identical hidden samples')
            indices=np.concatenate(record['indices'])
            if not np.array_equal(indices,base_indices):raise ValueError('Context comparison requires exact matching half/time/player indices')
            if args.save_paired_scores:
                score_arrays[f'{mode}/{condition}/error_m']=error
                score_arrays[f'{mode}/{condition}/indices']=indices
            summary=dict(model=summarize(error),paired_delta_mean_m=float((error-base).mean()),
                radius_80_coverage=float(np.concatenate(record['hit']).mean()),
                ball_input_displacement=summarize(np.concatenate(record['ball_error'])),
                by_age={name:summarize(error[mask]) for name,mask in [
                    ('up_to_3s',age<=3),('3_to_10s',(age>3)&(age<=10)),
                    ('10_to_30s',(age>10)&(age<=30)),('over_30s',(age>30)&np.isfinite(age)),
                    ('never_seen',~np.isfinite(age))]})
            results[mode]['conditions'][condition]=summary
            e=summary['model'];display.append([mode,condition,e['n'],round(e['mae_m'],3),
                round(e['median_m'],3),round(e['p90_m'],3),round(summary['paired_delta_mean_m'],3),
                f"{summary['radius_80_coverage']:.2%}",round(summary['ball_input_displacement']['mae_m'],3)])
    conditions=('Previously trained clean-control candidate; one held-out DFL match / two halves, '
        'fixed simulated broadcast masks and exact visible positions, IDs and roles. All conditions '
        'score exactly the same hidden player-timestamps. Ball is exact only in the control; stale '
        'ball takes the last exact sample every 30 s, using current/past samples only; midfield '
        'fallback never accesses ball observations. Unknown possession is -1 for both prior and '
        'learned features. Match phase and roster remain exact. These controlled missing-context '
        'stresses are not a fitted EPL error distribution or ball-detection evaluation. No new '
        'training, model selection, calibration, future hidden input, coordinate speed limiter, '
        'match update or promotion. Live remains causal and replay retrospective; old six-condition '
        'stress includes delayed-observation intersections with a smaller denominator and is not '
        'directly compared here. Radii exclude perception uncertainty and may be miscalibrated.')
    (output/'results.json').write_text(json.dumps(dict(match=match,results=results,conditions=conditions,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    if args.save_paired_scores:
        np.savez_compressed(output/'paired_scores.npz',**score_arrays)
    (output/'review.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<title>Missing ball and possession context</title><style>body{font:16px system-ui;max-width:1200px;'
        'margin:30px auto;padding:16px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #aaa;padding:8px}</style>'
        '<h1>Missing ball and possession context</h1><p>'+conditions+'</p>'+table(
        ['Mode','Input condition','Hidden samples','Mean m','Median m','P90 m','Paired mean change m',
         'Nominal 80% radius coverage','Mean ball-input displacement m'],display)+'</html>')
    print(display,flush=True)


if __name__=='__main__':main()
