"""Private before/after camera failure gallery; no reference truth or model fit."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from cv.session_report import table
from cv.soccernet import Calibration, DEFAULT_GAME


def project(h, foot):
    if h is None:return None
    p=np.asarray(h)@np.array([*foot,1.])
    return p[:2]/p[2] if abs(p[2])>1e-10 and np.isfinite(p).all() else None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',default='camera_continuity/appearance_raw');parser.add_argument('--out',default='camera_discontinuities')
    parser.add_argument('--rank-by',choices=['original','candidate'],default='original');args=parser.parse_args()
    root=Path('cv/out/audit8_20261011');output=root/args.out
    output.mkdir(parents=True,exist_ok=True)
    def load(name):return [json.loads(line) for line in (root/name/'detections.jsonl').open()]
    original=load('broadcast/appearance_raw');candidate=load(args.candidate)
    if len(original)!=len(candidate) or any(abs(a['t']-b['t'])>.001 for a,b in zip(original,candidate)):raise ValueError('Candidate frames do not align with source')
    frames=json.loads((root/'broadcast/frames.json').read_text())
    diagnostics=json.loads((root/args.candidate).parent.joinpath('camera_diagnostics.json').read_text())
    camera=Calibration(DEFAULT_GAME,1);cases=[]
    for i in range(1,len(original)):
        a,b=original[i-1:i+1];dt=b['t']-a['t']
        if not a['live'] or not b['live'] or abs(dt-.2)>.001:continue
        before={d['id']:d for d in a['dets'] if d['cls']=='person'}
        ca={d['id']:d for d in candidate[i-1]['dets'] if d['cls']=='person'}
        cb={d['id']:d for d in candidate[i]['dets'] if d['cls']=='person'}
        for d in b['dets']:
            old=before.get(d['id']) if d['cls']=='person' else None
            if old is None or old.get('shot_id')!=d.get('shot_id'):continue
            speed=float(np.hypot(d['px']-old['px'],d['py']-old['py'])/dt)
            foot=lambda box:np.array([(box[0]+box[2])/2,box[3]])
            now=foot(d.get('raw_box',d['box']));then=foot(old.get('raw_box',old['box']))
            hs=[camera.at(r['t'],allow_cross_shot=True) for r in [a,b]]
            points=[project(h,now) for h in hs]
            delta=float(np.linalg.norm(points[1]-points[0])) if all(p is not None for p in points) else None
            after=float(np.hypot(cb[d['id']]['px']-ca[d['id']]['px'],cb[d['id']]['py']-ca[d['id']]['py'])/dt) if d['id'] in ca and d['id'] in cb else None
            ranking=after if args.rank_by=='candidate' else speed
            if ranking is None or ranking<10:continue
            cases.append(dict(index=i,t0=a['t'],t1=b['t'],track=d['id'],original_speed_m_s=speed,candidate_speed_m_s=after,
                foot_displacement_pixels=float(np.linalg.norm(now-then)),same_foot_camera_change_m=delta,
                original_pitch=[[old['px'],old['py']],[d['px'],d['py']]],candidate_camera=diagnostics[i],
                feet=[then.tolist(),now.tolist()]))
    selected=[];bins=set()
    # Distinct times prevent one shared bad matrix filling the entire gallery.
    rank_key='candidate_speed_m_s' if args.rank_by=='candidate' else 'original_speed_m_s'
    for case in sorted(cases,key=lambda c:c[rank_key],reverse=True):
        key=round(case['t1']*2)
        if key in bins:continue
        selected.append(case);bins.add(key)
        if len(selected)==10:break
    cards=[];rows=[]
    for rank,c in enumerate(selected,1):
        panels=[]
        for offset,index in enumerate([c['index']-1,c['index']]):
            frame=frames[index];image=cv2.imread(str(root/'broadcast/images'/frame['image']))
            if image is None:raise RuntimeError(f'Missing review frame {frame}')
            foot=np.rint(c['feet'][offset]).astype(int);cv2.circle(image,tuple(foot),10,(0,255,255),3)
            panel=cv2.copyMakeBorder(cv2.resize(image,(640,360)),36,0,0,0,cv2.BORDER_CONSTANT,value=(24,24,24))
            cv2.putText(panel,f"{frame['t']:.1f}s / anonymous T{c['track']}",(12,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(245,245,245),1)
            panels.append(panel)
        path=f'jump_{rank:02d}.jpg';cv2.imwrite(str(output/path),np.vstack(panels))
        row=[rank,f"{c['t0']:.1f} → {c['t1']:.1f}",c['track'],round(c['foot_displacement_pixels'],2),round(c['same_foot_camera_change_m'],2) if c['same_foot_camera_change_m'] is not None else 'N/A',round(c['original_speed_m_s'],2),round(c['candidate_speed_m_s'],2) if c['candidate_speed_m_s'] is not None else 'N/A',c['candidate_camera']['status'],c['candidate_camera']['flow'].get('reason','supported ground flow')]
        rows.append(row);cards.append(f'<h2>Case {rank}: video {c["t1"]:.1f}s</h2><img src="{path}" alt="Consecutive actual video frames; yellow ring marks the same anonymous tracked foot">')
    conditions=(f'Ten largest {args.rank_by} projected discontinuities at distinct 2 Hz camera bins, using strictly consecutive same-shot anonymous measurements. '
        'Yellow rings mark measured image feet, not reference truth. Same-foot camera change isolates matrix changes while holding the current image point fixed. '
        f'Candidate missing support is N/A. Speeds are diagnostic artifacts, not verified physical motion. The gallery is selected by {args.rank_by} error symptoms, not an unbiased evaluation set. '
        'External cameras and shot labels are retrospective; no fitting, named roles or hidden inference. Keep broadcast images private.')
    (output/'results.json').write_text(json.dumps(dict(conditions=conditions,cases=selected),indent=2))
    (output/'review.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Private camera discontinuity gallery</title><style>body{font:16px system-ui;max-width:1250px;margin:30px auto;padding:16px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #aaa;padding:8px}img{max-width:100%}</style><h1>Remaining camera discontinuities</h1><p>'+conditions+'</p>'+table(['Case','Video seconds','Anonymous track','Foot motion px','Camera-only change m','Original m/s','Candidate m/s','Candidate camera','Ground-flow evidence'],rows)+''.join(cards)+'</html>')
    print(output/'review.html');print('Cases:',len(selected),'unresolved candidate over10:',sum(c['candidate_speed_m_s'] is not None and c['candidate_speed_m_s']>10 for c in selected))


if __name__=='__main__':main()
