"""Separate cached anonymous-track movement from role-assigned reconstruction.

Continuity is a diagnostic, never ground-truth positional accuracy.
"""
import json
from pathlib import Path

import numpy as np

from cv.session_report import table
from cv.soccernet import CALIB_FPS


def summary(values):
    x=np.asarray(values,float)
    return dict(n=len(x),median_m_s=float(np.median(x)) if len(x) else None,
        p90_m_s=float(np.percentile(x,90)) if len(x) else None,
        max_m_s=float(x.max()) if len(x) else None,steps_over_10m_s=int((x>10).sum()))


def measure(rows,step):
    selected=[r for r in rows if step<1 or abs(r['t']-round(r['t']))<.001]
    groups={name:[] for name in ('all','same_calibration_bin','new_calibration_bin')}
    for first,second in zip(selected,selected[1:]):
        dt=second['t']-first['t']
        if not first['live'] or not second['live'] or abs(dt-step)>.001:continue
        before={d['id']:d for d in first['dets'] if d['cls']=='person'}
        new=round(first['t']*CALIB_FPS)!=round(second['t']*CALIB_FPS)
        group='new_calibration_bin' if new else 'same_calibration_bin'
        for d in second['dets']:
            old=before.get(d['id']) if d['cls']=='person' else None
            if old is None or d.get('shot_id')!=old.get('shot_id'):continue
            speed=float(np.hypot(d['px']-old['px'],d['py']-old['py'])/dt)
            if not np.isfinite(speed):raise ValueError('Nonfinite cached measurement')
            groups['all'].append(speed);groups[group].append(speed)
    return {k:summary(v) for k,v in groups.items()}


def main():
    root=Path('cv/out/audit8_20261011');results={};display=[]
    for tracker in ('legacy_qualified','selected_qualified','appearance_raw'):
        rows=[json.loads(r) for r in (root/'broadcast'/tracker/'detections.jsonl').open()]
        results[tracker]={}
        for hz in (5,1):
            values=measure(rows,1/hz);results[tracker][str(hz)]=values
            for group,r in values.items():
                display.append([tracker,hz,group,r['n'],round(r['median_m_s'],2) if r['n'] else 'N/A',
                    round(r['p90_m_s'],2) if r['n'] else 'N/A',r['steps_over_10m_s'],round(r['max_m_s'],2) if r['n'] else 'N/A'])
    conditions=('Original frozen CCBV-based projected feet, shot-qualified anonymous IDs, '
        'strictly consecutive supported measurements. No formation, hidden model or named role '
        'assignment. A 10 m/s flag is a diagnostic threshold, not a tracking-accuracy label. '
        'Camera-bin changes reflect nearest 2Hz calibration sampling and correlate with time '
        'during camera motion; this grouping alone cannot identify calibration as the cause. '
        'Different trackers/Hz have different matched-transition sets. Active exports are untouched.')
    output=root/'anonymous_continuity';output.mkdir(parents=True,exist_ok=True)
    (output/'results.json').write_text(json.dumps(dict(results=results,conditions=conditions),indent=2))
    (output/'review.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Anonymous measurement movement</title><style>body{font:16px system-ui;max-width:1250px;margin:30px auto;padding:16px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #aaa;padding:8px}</style><h1>Anonymous measurements before role assignment</h1><p>'+conditions+'</p>'+table(['Tracker','Hz','Camera bin','Transitions','Median m/s','P90 m/s','Steps >10 m/s','Maximum m/s'],display)+'</html>')
    print(results,flush=True)


if __name__=='__main__':main()
