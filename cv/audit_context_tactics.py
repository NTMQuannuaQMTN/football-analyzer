"""Bounded diagnostic geometry scoring after frozen context-validation gates."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from cv.session_report import table, display_rows


def render(root,records):
    rows=[];paired={};counts={};status=[]
    for fold in records:
        status.append([fold['fold'],fold['test'],fold['variant'],fold['status']])
        path=root/f"fold_{fold['fold']}"/fold['variant']/'results.json'
        if fold['status']!='completed':continue
        result=json.loads(path.read_text())
        if result['match']!=fold['test']:raise ValueError('Tactical scoring match is not the excluded test')
        for mode,modes in result['results'].items():
            for condition,metrics in modes['conditions'].items():
                for metric,r in metrics.items():
                    key=(fold['fold'],mode,condition,metric)
                    paired.setdefault(key,{})[fold['variant']]=r
                    rows.append([fold['fold'],fold['test'],fold['variant'],mode,condition,metric,r['n'],
                        round(r['mean_m'],3),round(r['median_m'],3),round(r['p90_m'],3),
                        f"{r['sensitivity_envelope_coverage']:.2%}",round(r['mean_envelope_span_m'],2)])
    groups={}
    for (_,mode,condition,metric),variants in paired.items():
        if len(variants)!=2:continue
        a,b=variants['clean_control'],variants['robust']
        if a['n']!=b['n']:raise ValueError('Tactical geometry sample counts differ')
        key=(mode,condition,metric)
        groups.setdefault(key,[]).append([a['mean_m'],b['mean_m']])
        counts[key]=counts.get(key,0)+a['n']
    macro=[]
    for key,values in groups.items():
        a,b=np.mean(values,axis=0)
        macro.append([*key,len(values),counts[key],round(a,3),round(b,3),round(b-a,3)])
    scope=('Previously reviewed German matches, fresh nested-match V3 training; only folds with both '
        'frozen validation gates and completed equal-budget fits are scored. Candidate weights remain '
        'private. Exact visible positions/IDs/roles/possession/phase and simulated broadcast masks. '
        'Ball every30s means past/current exact samples held, not a measured EPL detector error rate. '
        'At least seven active outfield players and one inferred player per frame-team. Hidden truth '
        'enters scoring only after prediction. Width/depth and four-deepest-player line use existing '
        'production definitions; raw predictions have no event cue or speed limiter. Live is causal, '
        'replay retrospective. Equal-match mean errors have separate per-match sample counts; frame-team '
        'observations are correlated. Radius bounds are geometric sensitivity envelopes, not simultaneous '
        '80% confidence intervals; observed-player perception error is excluded. No tactical intent or '
        'causal outcome is scored, and no promotion, calibration or parameter selection uses these results.')
    summary=dict(records=records,macro_rows=macro,rows=rows,conditions=scope,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/'results.json').write_text(json.dumps(summary,indent=2))
    (root/'review.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<title>Context error and tactical geometry</title><style>body{font:16px system-ui;'
        'max-width:1500px;margin:30px auto;padding:16px}table{border-collapse:collapse;width:100%;'
        'margin-bottom:32px}td,th{border:1px solid #aaa;padding:7px}</style>'
        '<h1>Context error and tactical geometry</h1><p>'+scope+'</p><h2>Executed checks</h2>'
        +table(['Fold','Excluded match','Variant','Status'],display_rows(status))+'<h2>Equal-match mean errors</h2>'
        +table(['Mode','Context','Geometry feature','Paired matches','Frame-team samples',
        'Control mean m','Candidate mean m','Change m'],display_rows(macro))+'<h2>Per-match errors and bounds</h2>'
        +table(['Fold','Match','Variant','Mode','Context','Feature','Frame-team samples','Mean m',
        'Median m','P90 m','Reference in sensitivity envelope','Mean envelope span m'],display_rows(rows))+'</html>')
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--training',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--deadline',type=float,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True);records=[]
    folds=json.loads((args.training/'results.json').read_text())
    for fold in folds:
        if not all(fold['mode_validation_gates'].values()):continue
        training=fold['training']
        if len(training)!=4 or any(len(r['history'])!=30 or not all(h['complete_training_epoch'] for h in r['history']) for r in training):
            raise ValueError('Thirty full training epochs per model are required')
        for mode in ['live','replay']:
            if len({r['total_optimizer_updates'] for r in training if r['mode']==mode})!=1:
                raise ValueError('Control/candidate update budgets differ')
        for variant in ['clean_control','robust']:
            if args.deadline-time.time()<30:break
            output=args.out/f"fold_{fold['fold']}"/variant;output.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,'-m','cv.audit_tactical_sensitivity','--match',fold['test'],
                '--live-checkpoint',str(args.training/f"fold_{fold['fold']}"/f'hidden_live_robustness_{variant}_v3.pt'),
                '--replay-checkpoint',str(args.training/f"fold_{fold['fold']}"/f'hidden_replay_robustness_{variant}_v3.pt'),
                '--ball-context-stress','--out',str(output)]
            record=dict(fold=fold['fold'],test=fold['test'],variant=variant,command=command)
            previous=output/'results.json'
            if previous.is_file():
                cached=json.loads(previous.read_text())
                current_hash=hashlib.sha256(Path('cv/audit_tactical_sensitivity.py').read_bytes()).hexdigest()
                if cached['match']==fold['test'] and cached['source_sha256']==current_hash:
                    record.update(status='completed',exit_code=0,reused_completed_identical_source=True)
                    records.append(record);render(args.out,records);print(record,flush=True);continue
            with (output/'execution.log').open('w') as log:
                try:
                    result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,args.deadline-time.time()))
                    record.update(status='completed' if result.returncode==0 else 'failed',exit_code=result.returncode)
                except subprocess.TimeoutExpired:record.update(status='not completed: deadline')
            records.append(record);render(args.out,records)
            print(record,flush=True)
    (args.out/'finished.json').write_text(json.dumps(dict(finished_epoch=time.time(),deadline=args.deadline,
        planned_fold_count=len(folds),executed_model_checks=len(records))))


if __name__=='__main__':main()
