"""Oracle-visible ablation isolates role assignment from video perception.

Exact visible position/track/team labels enter this explicitly diagnostic
ablation. It is not a video inference benchmark or verified shirt identity.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from cv.audit_tracking import identity_counts
from cv.gsr import clips,load_clip
from cv.roles import RoleEstimator,lineup
from cv.soccernet import DEFAULT_GAME


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--part',choices=['val','test'],required=True);a=ap.parse_args();torch.set_num_threads(1)
    games=['3','5'] if a.part=='val' else ['2'];totals={k:Counter() for k in ['legacy','stable_track']};cases=[]
    templates={'A':lineup(DEFAULT_GAME,'home')[0],'B':lineup(DEFAULT_GAME,'away')[0]}
    for clip in clips('valid'):
        info,frames=load_clip(clip)
        if info['game_id'] not in games:continue
        estimators={mode:{team:RoleEstimator(team,roles,mode) for team,roles in templates.items()} for mode in totals}
        counts={k:Counter() for k in totals};gt_count=Counter();pred_count={k:Counter() for k in totals};overlaps={k:Counter() for k in totals};previous={k:{} for k in totals}
        ball=np.array([52.5,34.]);positions={k:{} for k in totals}
        for second,frame in enumerate(frames[::25]):
            players=[p for p in frame['people'] if p['role'] in ('player','goalkeeper') and p['team'] in ('left','right') and p['pitch'] is not None]
            visible_ball=[p['pitch'] for p in frame['people'] if p['role']=='ball' and p['pitch'] is not None]
            if visible_ball:ball=np.array(visible_ball[0])
            for team,label in [('A','left'),('B','right')]:
                own=[p for p in players if p['team']==label];opponents=[p['pitch'] for p in players if p['team']!=label]
                gt_count.update((team,p['track']) for p in own)
                dets=[(p['track'],np.array(p['pitch']),'GK' if p['role']=='goalkeeper' else 'OUT') for p in own]
                for mode,est in estimators.items():
                    pos,obs=est[team].step(second,ball,0,dets,opponents);c=counts[mode];c['gt']+=len(own);c['observed']+=int(obs.sum())
                    for sid,role in est[team].last_assignments.items():
                        g=(team,sid);p=(team,role);pred_count[mode][p]+=1;overlaps[mode][g,p]+=1
                        if g in previous[mode]:c['transitions']+=1;c['role_switches']+=previous[mode][g]!=p
                        previous[mode][g]=p
                        old=positions[mode].get(p)
                        if old is not None and old[0]==second-1:
                            c['consecutive_observed']+=1;c['observed_steps_over_10m_s']+=int(np.linalg.norm(pos[role]-old[1])>10)
                        positions[mode][p]=(second,pos[role].copy())
        for mode in totals:
            counts[mode].update(identity_counts(gt_count,pred_count[mode],overlaps[mode]));totals[mode].update(counts[mode])
        cases.append(dict(clip=clip.name,game=info['game_id'],results={k:dict(v) for k,v in counts.items()}));print(a.part,clip.name,flush=True)
    result={}
    for k,v in totals.items():
        result[k]={**v,'coverage':v['observed']/max(1,v['gt']),'anonymous_role_identity_f1':2*v['idtp']/max(1,2*v['idtp']+v['idfp']+v['idfn'])}
    output=dict(part=a.part,results=result,per_clip=cases,
        conditions='Oracle visible labels provide exact positions, track IDs and teams; current visible ball only (held last ball when absent). Fixed Arsenal/Liverpool role templates for anonymous aliases, no verified naming. 1Hz causal assignment; global clip mapping in evaluation only. Hidden states and video perception not evaluated.')
    root=Path('cv/out/audit8_20261011/role_aliases');root.mkdir(parents=True,exist_ok=True);(root/f'{a.part}.json').write_text(json.dumps(output,indent=2));print(result,flush=True)


if __name__=='__main__':main()
