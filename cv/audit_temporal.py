"""Controlled timestamp ablation: exact visible coordinates and identities.

This isolates bucket-averaging bias. It is not reconstruction accuracy and
the zero nearest-sample error with exact inputs is not a model improvement.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from cv.gsr import clips,load_clip
from cv.temporal_samples import sample_seconds


def main():
    output={}
    for part,games in [('val',['3','5']),('test',['2'])]:
        errors=defaultdict(list);counts=defaultdict(int);offsets=defaultdict(list);per_clip=[]
        for clip in clips('valid'):
            info,frames=load_clip(clip)
            if info['game_id'] not in games:continue
            rows=defaultdict(list);references={}
            for index,f in enumerate(frames[::5]):
                t=index*.2
                for p in f['people']:
                    if p['pitch'] is None or p['role'] not in ('player','goalkeeper','referee'):continue
                    rows[p['track']].append((t,*p['pitch']))
                    if index%5==0:references[(int(round(t)),p['track'])]=p['pitch']
            local=defaultdict(list)
            for track,samples in rows.items():
                a,at=sample_seconds(samples,29);b,bt=sample_seconds(samples,29,'nearest')
                for sec in sorted(set(a)&set(b)):
                    reference=references.get((sec,track))
                    if reference is None:continue
                    for name,p,t in [('bucket_mean',a[sec],at[sec]),('nearest',b[sec],bt[sec])]:
                        error=float(np.linalg.norm(p-reference));errors[name].append(error);local[name].append(error);offsets[name].append(t-sec)
            per_clip.append(dict(clip=clip.name,game=info['game_id'],n=len(local['nearest']),mean_bucket_bias_m=float(np.mean(local['bucket_mean'])) if local['bucket_mean'] else None))
        metrics={name:dict(n=len(e),mean_m=float(np.mean(e)),median_m=float(np.median(e)),p90_m=float(np.percentile(e,90)),mean_source_time_offset_s=float(np.mean(offsets[name]))) for name,e in errors.items()}
        output[part]=dict(metrics=metrics,per_clip=per_clip,
            conditions='Exact labelled visible pitch coordinates and exact identities sampled at 5Hz. Scoring at integer timestamps on common visible references. Measures temporal aggregation bias only; no detector, camera or hidden-player accuracy claim.')
    path=Path('cv/out/audit8_20261011/temporal_bias.json');path.write_text(json.dumps(output,indent=2));print({p:r['metrics'] for p,r in output.items()})


if __name__=='__main__':main()
