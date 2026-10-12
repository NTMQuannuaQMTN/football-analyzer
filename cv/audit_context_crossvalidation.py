"""Render executed nested-match context experiments without selecting models."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cv.audit_ball_conditioning import CONDITIONS
from cv.session_report import table, display_rows


def generate(root):
    config=json.loads((root/'config.json').read_text())
    records=json.loads((root/'results.json').read_text())
    gates=[];rows=[];age_rows=[];pairs={};counts={};sources={};complete=0
    for fold in records:
        train,val,test=fold['train'],fold['val'],fold['test']
        if len(set(train+[val,test]))!=7 or val in train or test in train or val==test:
            raise ValueError('Whole-match train/validation/test partitions overlap')
        training=fold['training']
        finished=len(training)==4 and all(len(r['history'])==30 and all(h['complete_training_epoch'] for h in r['history']) for r in training)
        equal=finished and all(len({r['total_optimizer_updates'] for r in training if r['mode']==mode})==1 for mode in ['live','replay'])
        complete+=bool(finished)
        gates.append([fold['fold'],test,val,len(train),'yes' if finished else 'no',
            'yes' if equal else 'no',str(fold['mode_validation_gates'].get('live',False)),
            str(fold['mode_validation_gates'].get('replay',False)),fold['held_out_candidate_status']])
        loaded={}
        for variant in ['clean_control','robust']:
            path=root/f"fold_{fold['fold']}"/f'evaluation_{variant}'/'results.json'
            execution=next((e for e in fold['evaluations'] if e['variant']==variant),{})
            if not path.is_file() or execution.get('exit_code')!=0:continue
            if variant=='robust' and not (equal and all(fold['mode_validation_gates'].values())):
                raise ValueError('Candidate evaluation exists without both equal-budget validation gates')
            result=json.loads(path.read_text())
            if result['match']!=test:raise ValueError('Evaluation match differs from excluded match')
            loaded[variant]=result
            sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        if len(loaded)!=2:continue
        for mode in ['live','replay']:
            for condition in CONDITIONS:
                a=loaded['clean_control']['results'][mode]['conditions'][condition]
                b=loaded['robust']['results'][mode]['conditions'][condition]
                if a['model']['n']!=b['model']['n']:raise ValueError('Candidate sample counts differ')
                key=(mode,condition);n=a['model']['n']
                delta=b['model']['mae_m']-a['model']['mae_m']
                pairs.setdefault(key,[]).append((a['model']['mae_m'],b['model']['mae_m']))
                counts[key]=counts.get(key,0)+n
                rows.append([fold['fold'],test,mode,condition,n,a['model']['mae_m'],b['model']['mae_m'],
                    round(delta,2),round(a['model']['p90_m'],2),round(b['model']['p90_m'],2),
                    f"{a['radius_80_coverage']:.2%}",f"{b['radius_80_coverage']:.2%}"])
                if condition in ['exact_context','ball_observed_every_30s']:
                    for age,control in a['by_age'].items():
                        candidate=b['by_age'][age]
                        if control['n']!=candidate['n']:raise ValueError('Age-group sample counts differ')
                        if not control['n']:continue
                        age_rows.append([fold['fold'],test,mode,condition,age,control['n'],
                            control['mae_m'],candidate['mae_m'],round(candidate['mae_m']-control['mae_m'],2),
                            round(control['p90_m'],2),round(candidate['p90_m'],2)])
    macro=[]
    for (mode,condition),values in pairs.items():
        means=np.mean(values,axis=0)
        macro.append([mode,condition,len(values),counts[mode,condition],round(means[0],3),
            round(means[1],3),round(means[1]-means[0],3)])
    scope=('Fresh V3 models, seed11026 plus fold index; five training matches, one separate '
        'validation match and one excluded test match. Thirty complete epochs and equal optimiser '
        'updates per variant/mode are required. Validation alone gates candidate testing; failed '
        'or incomplete folds are not silently included in the paired summary. Exact visible '
        'positions/IDs/roles/phase, simulated camera masks, seven previously reviewed German '
        'matches. These are not EPL video results. Means in individual evaluation artifacts are '
        'rounded to .01m; changes below that resolution are not interpreted. Macro means give '
        'each successfully paired match equal weight, not each player-timestamp. P90 is shown '
        'per match, never averaged as a pooled percentile. Live ball holds use past/current '
        'samples; replay may use later player observations. Nominal80% radius coverage excludes '
        'real-video perception error. No model or export is promoted by this report.')
    source=root/'results.json';sources[str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
    summary=dict(attempted_folds=len(records),complete_training_folds=complete,
        planned_folds=len(config['matches']),gate_rows=gates,paired_rows=rows,macro_rows=macro,age_rows=age_rows,
        conditions=scope,source_sha256=sources)
    (root/'review_results.json').write_text(json.dumps(summary,indent=2))
    (root/'review.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<title>Nested-match context training comparison</title><style>body{font:16px system-ui;'
        'max-width:1500px;margin:30px auto;padding:16px}table{border-collapse:collapse;width:100%;'
        'margin-bottom:32px}td,th{border:1px solid #aaa;padding:7px}nav a{margin-right:16px}</style>'
        '<h1>Nested-match context training comparison</h1><p>'+scope+'</p>'
        f'<p>{len(records)} attempted / {complete} complete training folds of {len(config["matches"])} planned.</p>'
        '<h2>Validation gates and execution</h2>'+table(['Fold','Excluded test','Separate validation',
        'Training matches','30 complete epochs ×4','Equal updates by mode','Live gate','Replay gate',
        'Candidate test status'],gates)+'<h2>Equal-match means, successfully paired folds only</h2>'
        +table(['Mode','Context','Paired matches','Hidden samples','Control mean m','Candidate mean m',
        'Candidate minus control m'],display_rows(macro))+'<h2>Per-match results</h2>'+table(['Fold','Excluded match',
        'Mode','Context','Hidden samples','Control mean m','Candidate mean m','Change m','Control p90 m',
        'Candidate p90 m','Control radius coverage','Candidate radius coverage'],display_rows(rows))
        +'<h2>Error by time since last observation</h2><p>Per-match counts; never seen is separate '
        'from a finite last-observation interval. Replay can use subsequent observations.</p>'
        +table(['Fold','Excluded match','Mode','Context','Last-observation age','Hidden samples',
        'Control mean m','Candidate mean m','Change m','Control p90 m','Candidate p90 m'],display_rows(age_rows))+'</html>')
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();summary=generate(args.out)
    print('attempted',summary['attempted_folds'],'complete',summary['complete_training_folds'],
          'macro',summary['macro_rows'],flush=True)


if __name__=='__main__':main()
