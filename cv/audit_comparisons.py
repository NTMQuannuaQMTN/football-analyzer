"""Paired differences and clip bootstrap diagnostics, with limited scope."""
import json
from pathlib import Path

import numpy as np

from cv.audit_tracking import summary


def interval(values):
    return [float(v) for v in np.percentile(values,[2.5,97.5])]


def main():
    root=Path('cv/out/audit8_20261011');rng=np.random.default_rng(11026)
    tracking=json.loads((root/'tracking_generic/test.json').read_text());rows=tracking['per_clip'];metrics=['recall','precision','identity_f1'];deltas={k:[] for k in metrics}
    keys=['gt','pred','matched','idtp','idfn','idfp','switches','transitions']
    for _ in range(2000):
        chosen=rng.integers(len(rows),size=len(rows));totals={k:{key:sum(rows[i]['results'][k][key] for i in chosen) for key in keys} for k in ['legacy','selected']}
        score={k:summary(v) for k,v in totals.items()}
        for m in metrics:deltas[m].append(score['selected'][m]-score['legacy'][m])
    tracker=dict(clip_units=len(rows),metrics={k:dict(delta=tracking['results']['selected'][k]-tracking['results']['legacy'][k],clip_bootstrap_95_interval=interval(v)) for k,v in deltas.items()})
    source=json.loads((root/'camera_motion/test.json').read_text());indexed={(r['clip'],r['image'],r['configuration']):r for r in source['cases']};common=[];added=[];removed=[]
    for r in source['cases']:
        if r['configuration']!='selected':continue
        old=indexed[r['clip'],r['image'],'baseline']
        if r['supported'] and old['supported']:common.append(dict(clip=r['clip'],before=old['errors_m'],after=r['errors_m']))
        elif r['supported']:added.extend(r['errors_m'])
        elif old['supported']:removed.extend(old['errors_m'])
    before=np.array([e for r in common for e in r['before']]);after=np.array([e for r in common for e in r['after']]);clips=sorted({r['clip'] for r in common})
    differences=[]
    grouped={c:[r for r in common if r['clip']==c] for c in clips}
    for _ in range(2000):
        chosen=rng.choice(clips,size=len(clips),replace=True);b=[];a=[]
        for c in chosen:
            for r in grouped[c]:b.extend(r['before']);a.extend(r['after'])
        differences.append(float(np.mean(a)-np.mean(b)))
    camera=dict(common_frames=len(common),common_people=len(before),common_before_mean_m=float(before.mean()),common_after_mean_m=float(after.mean()),
        mean_delta_95_clip_bootstrap=interval(differences),additional_people=len(added),additional_mean_m=float(np.mean(added)) if added else None,removed_people=len(removed),
        note='Coverage-changing aggregate means cannot be treated as paired improvement; this common-set comparison keeps the same people/frame samples.')
    result=dict(tracker=tracker,camera=camera,seed=11026,replicates=2000,
        conditions='Resample whole clips to preserve within-clip temporal/player correlation. All test clips are from ONE game; clips may remain correlated and this does not quantify new-match/EPL generalisation. Test intervals do not select configurations.')
    (root/'paired_comparisons.json').write_text(json.dumps(result,indent=2));print(result,flush=True)


if __name__=='__main__':main()
