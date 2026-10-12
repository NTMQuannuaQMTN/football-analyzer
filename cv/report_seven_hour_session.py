"""Render completed session artifacts; never infer completion or accuracy from logs."""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from cv.session_report import table


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def generate(root):
    session=read(root/'session.json');finished=read(root/'supervisor_finished.json')
    lines=['# Seven-hour application — recorded results','',
        'Generated '+datetime.now(ZoneInfo('Asia/Singapore')).isoformat(timespec='seconds'),'',
        'Supervisor ended: '+str(bool(finished))+'. This is an artifact-based report; missing stages remain incomplete.',
        '', 'DFL uses exact visible tracking and simulated visibility. EPL hidden truth is missing. No broadcast accuracy is inferred from DFL error. Live uses past sightings; replay may use later sightings. Legacy event postprocessing differs. No production weights are promoted.', '']
    parts=['<h1>Recorded match accuracy</h1><p>DFL tracking with simulated visibility; not broadcast perception. EPL hidden truth N/A. Each row is a separate match-half and mode; sample counts are correlated player-timestamps. Models selected using validation, not these test errors.</p>']
    for phase in ('preview','trained'):
        records=read(root/f'dfl_{phase}_metrics.json') or [];rows=[]
        for r in records:
            for mode,m in r['metrics'].items():
                x=m['hidden'];old=r.get('baseline');b=old[mode]['hidden']['mean_m'] if old else None
                rows.append([r['match'],r['half'],mode,x['n'],round(x['mean_m'],3),round(x['median_m'],3),round(x['p90_m'],3),round(b,3) if b is not None else None,m['observed_coordinates_preserved']])
        parts.append(f'<h2>{phase}: {len(records)}/14 DFL halves</h2>'+table(['Match','Half','Mode','Hidden N','Mean m','Median m','P90 m','Legacy mean m','Observed preserved'],rows))
        lines.extend([f'## {phase}: {len(records)}/14 DFL halves','', '| Match | Half | Mode | Hidden N | Mean m | Median m | P90 m | Legacy mean m | Observed preserved |','|---|---:|---|---:|---:|---:|---:|---:|---|'])
        for row in rows:lines.append('| '+' | '.join(str(v) if v is not None else 'N/A' for v in row)+' |')
        lines.append('')
        for mode in ('live','replay'):
            cells=[r['metrics'][mode]['hidden'] for r in records];n=sum(c['n'] for c in cells)
            if n:lines.append(f'{mode}: hidden N={n:,}, sample-weighted mean={sum(c["n"]*c["mean_m"] for c in cells)/n:.3f} m. Overall median/P90 not computed by averaging row quantiles.')
        applied=read(root/f'{phase}_applied.json');other=read(root/f'other_epl_{phase}.json')
        lines.extend(['',f'EPL application marker: {bool(applied)}; other EPL half exports: {len(other["applied_files"]) if other else 0}/4. Hidden accuracy: N/A. Named identities and video calibration remain unverified.',''])
    training=read(root/'dfl_training/finished.json')
    lines.extend(['## Training completion','',f'DFL complete folds: {training.get("complete_training_folds",0) if training else 0}/7. Trainer finished: {bool(training)}.',''])
    gpu=read(root/'gpu_execution.json') or []
    for row in gpu:
        lines.append(f'- {row["stage"]}: exit code {row["exit_code"]}, elapsed {(row["finished_epoch"]-row["started_epoch"])/60:.1f} min. Exit code zero alone is not an accuracy gate.')
    if not gpu:lines.append('No completed GPU stage record yet.')
    camera=read(root/'camera_evaluation/paired_evaluation/results.json')
    if camera:
        rows=[]
        for name,value in camera['summary'].items():
            x=value['common_supported'];rows.append([name,camera['common_frames'],x['n'],round(x['mean_m'],3),round(x['p90_m'],3),value['gate_passed']])
        parts.append('<h2>Frozen marking-camera validation</h2><p>Matched supported frames only. Visible labelled feet; not hidden-player or identity accuracy. Support failures are recorded separately in the camera review.</p>'+table(['Camera','Common frames','Feet N','Mean m','P90 m','Validation gate'],rows))
        lines.extend(['','## Camera validation','', '| Camera | Common frames | Feet N | Mean m | P90 m | Gate |','|---|---:|---:|---:|---:|---|'])
        lines.extend('| '+' | '.join(map(str,row))+' |' for row in rows)
        lines.extend(['','Full support and user-landmark results: `camera_evaluation/paired_evaluation/review.html`. No camera test/promotion after a failed validation gate.'])
    for phase in ('preview','trained'):
        verification=read(root/f'demo_{phase}_verification.json')
        if verification:lines.extend(['',f'{phase} demo: health, {verification["candidate_cards"]} candidate cards, Arsenal API and studio HTTP route verified. Rendered-browser/manual UI review remains required.'])
    lines.extend(['',
        'The running DFL config retained an old prose phrase “Thirty complete epochs”; actual eligibility checks use the requested150 epochs and equal updates. Source snapshots/configs are retained. This wording discrepancy does not relax the gate.',''])
    errors=read(root/'application_failures.json') or []
    lines.extend(['## Failures and limitations','']+[f'- Attempt {e["attempt"]}: {e["cause"]}. Handling: {e.get("fixed","N/A")}.' for e in errors])
    for marker in ('final_application_blocker.json','closeout_checks.json'):
        value=read(root/marker)
        if value:lines.extend(['',marker+': '+str(value)])
    preserved=[]
    for manifest in ('preservation_final.json','active_exports_preservation_final.json'):
        original=read(Path('cv/out/audit8_20261011')/manifest)
        for row in original['rows']:
            p=Path(row.get('path',row.get('file','')));same=p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
            preserved.append({'file':str(p),'unchanged':same})
    failures=[r for r in preserved if not r['unchanged']]
    (root/'originals_preservation.json').write_text(json.dumps({'checks':len(preserved),'failures':failures,'all_unchanged':not failures},indent=2))
    lines.extend(['','## Original-file preservation','',f'{len(preserved)} manifest checks (overlap), failures: {len(failures)}. Detailed paths in originals_preservation.json.','',
        'Camera/appearance/jersey experiments are not automatically deployed. Unsupported close-up/set-piece angles must abstain. DFL exact IDs/roles/context do not exist reliably in the EPL pipeline. No inferred positions were treated as ground truth. More synchronized licensed EPL video/tracking remains an optional user input.'])
    (root/'results.md').write_text('\n'.join(lines)+'\n')
    (root/'results.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Recorded accuracy</title><style>body{font:16px system-ui;margin:30px;max-width:1400px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #aaa}</style>'+''.join(parts)+'</html>')
    if finished:
        report=Path('docs/SEVEN_HOUR_APPLICATION_20261012.md')
        if report.exists():
            report.write_text(report.read_text().replace('Status: running. This is not a completed training report or an accuracy improvement claim.',
                'Status: session ended. See recorded closeout results below for completed and incomplete stages. No improvement is assumed.'))
    return root/'results.md'


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args();print(generate(args.root))
