"""Explain the frozen full-validation camera gate using identical reference feet.

Evaluation-only: groups missing/new/common support, never chooses parameters.
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from cv.audit_visible_pipeline import stats
from cv.session_report import table


def main():
    root=Path('cv/out/audit8_20261011');source=root/'dense_foreground_full_validation/val.json'
    data=json.loads(source.read_text());out=root/'camera_coverage_tradeoffs';out.mkdir(exist_ok=True)
    groups={name:{method:[] for method in ['strict','foreground_exclusion']}
            for name in ['common_supported','new_candidate_support','lost_candidate_support']}
    clips=defaultdict(lambda:{name:[] for name in ['strict','foreground_exclusion','paired_delta']})
    changed=[];counts=defaultdict(int)
    for case in data['cases']:
        a=case['methods']['strict'];b=case['methods']['foreground_exclusion']
        category='common_supported' if a['supported'] and b['supported'] else (
            'new_candidate_support' if b['supported'] else 'lost_candidate_support' if a['supported'] else None)
        if category:
            counts[category]+=1
            for name,record in [('strict',a),('foreground_exclusion',b)]:
                groups[category][name].extend(record['errors_m'])
        for name,record in [('strict',a),('foreground_exclusion',b)]:
            clips[case['clip']][name].extend(record['errors_m'])
        if a['supported'] and b['supported']:
            if len(a['errors_m'])!=len(b['errors_m']):raise ValueError('Paired feet must preserve exact reference order')
            clips[case['clip']]['paired_delta'].extend(np.array(b['errors_m'])-a['errors_m'])
        if category in ['new_candidate_support','lost_candidate_support']:
            changed.append(dict(clip=case['clip'],frame=case['frame'],t=case['t'],category=category,
                reference_people=case['people'],strict=stats(a['errors_m']),foreground=stats(b['errors_m'])))
    summary={category:dict(frames=counts[category],methods={name:stats(values) for name,values in methods.items()})
             for category,methods in groups.items()}
    per_clip={clip:{name:stats(values) for name,values in methods.items()} for clip,methods in clips.items()}
    conditions=('All forty validation clips, same 6,000 timestamps / 101,388 pitch-labelled people; '
        'frozen camera parameters. Exact labelled image feet score only after cameras are fixed. '
        'Common support pairs the same reference person order; new/lost support is reported separately. '
        'Per-clip errors weight that clip\'s scored feet. Changed-support examples are diagnostic '
        'selections, not unbiased accuracy estimates or a basis for threshold reselection. '
        'No model, cache, active export or original gate is changed.')
    (out/'results.json').write_text(json.dumps(dict(groups=summary,per_clip=per_clip,changed_support=changed,
        conditions=conditions,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()),indent=2))
    rows=[]
    for category,r in summary.items():
        for method,e in r['methods'].items():
            rows.append([category,r['frames'],method,e['n'],e['mean_m'],e['median_m'],e['p90_m']])
    clip_rows=[]
    for clip,r in sorted(per_clip.items(),key=lambda pair:pair[1]['paired_delta']['mean_m'] or 0,reverse=True):
        a,b,d=r['strict'],r['foreground_exclusion'],r['paired_delta']
        clip_rows.append([clip,a['n'],a['mean_m'],a['p90_m'],b['n'],b['mean_m'],b['p90_m'],d['n'],d['mean_m']])
    (out/'review.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<title>Camera coverage and error trade-offs</title><style>body{font:16px system-ui;max-width:1300px;'
        'margin:30px auto;padding:16px}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{border:1px solid #aaa;padding:8px}</style>'
        '<h1>Camera coverage and error trade-offs</h1><p>'+conditions+'</p>'+table(
        ['Support group','Frames','Camera','Reference feet','Mean m','Median m','P90 m'],rows)+
        '<h2>Per-clip errors, sorted by paired mean regression</h2>'+table(
        ['Clip','Strict feet','Strict mean m','Strict p90 m','Candidate feet','Candidate mean m',
         'Candidate p90 m','Paired feet','Paired mean change m'],clip_rows)+'</html>')
    print(summary,flush=True)


if __name__=='__main__':main()
