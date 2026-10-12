"""Link human shirt clicks to anonymous image tracks, without fitting identities."""
import argparse
import hashlib
from html import escape
import json
from pathlib import Path

import numpy as np

from cv.evaluate_manual_camera import read_labels
from cv.session_report import table


def associate(label,row,max_distance=20,ambiguity_gap=6,point_kind='foot'):
    """Conservative image-foot association; no pitch coordinates or formation."""
    if row is None or not row['live']:
        return dict(status='N/A',reason='Exact supported measurement missing',track=None)
    click=np.array([label['pixel_x'],label['pixel_y']],float)
    candidates=[]
    for detection in [*row['dets'],*row.get('unassigned_dets',[])]:
        if detection.get('cls')!='person':continue
        box=detection.get('raw_box',detection['box'])
        if point_kind=='body':
            if box[0]<=click[0]<=box[2] and box[1]<=click[1]<=box[3]:
                candidates.append(detection)
            continue
        foot=np.array([(box[0]+box[2])/2,box[3]])
        candidates.append((float(np.linalg.norm(foot-click)),detection))
    if point_kind=='body':
        if len(candidates)!=1:
            return dict(status='N/A',reason='No containing person box' if not candidates else 'Overlapping person boxes; identity ambiguous',track=None)
        track=candidates[0].get('id')
        return dict(status='anonymous box linked' if track is not None else 'detection unconfirmed',track=track,
                    reason='Body click identifies an image box only; no ground-contact point, pitch position or automatic shirt recognition verified')
    candidates.sort(key=lambda item:item[0])
    if not candidates or candidates[0][0]>max_distance:
        return dict(status='N/A',reason='No detected foot within 20 image pixels',track=None)
    if len(candidates)>1 and candidates[1][0]-candidates[0][0]<ambiguity_gap:
        return dict(status='N/A',reason='Nearby detected feet make the click ambiguous',track=None)
    distance,detection=candidates[0]
    track=detection.get('id')
    return dict(status='anonymous track linked' if track is not None else 'detection unconfirmed',
                reason='Human label does not verify automatic shirt recognition or pitch calibration',
                track=track,distance_pixels=distance)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--csv',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,default=Path('cv/out/audit8_20261011/broadcast/appearance_raw'))
    parser.add_argument('--root',type=Path,default=Path('cv/out/audit8_20261011/manual_review'))
    parser.add_argument('--basis',choices=['human','assistant_provisional'],default='human')
    parser.add_argument('--point-kind',choices=['foot','body'],default='foot',help='Body clicks link a unique containing box only, never a ground-contact reference')
    parser.add_argument('--name',default='manual_identity_links',help='Output basename; retain foot and body reviews separately')
    args=parser.parse_args()
    if Path(args.name).name!=args.name:raise ValueError('Output name must be a basename')
    frames=json.loads((args.root/'frames.json').read_text())
    labels=read_labels(args.csv,frames,args.basis)
    run=json.loads((args.run_dir/'run.json').read_text())
    if run['game']!='england_epl/2016-2017/2016-08-14 - 18-00 Arsenal 3 - 4 Liverpool' or run['half']!=1:
        raise ValueError('This annotation page is specific to the Arsenal first-half clip')
    rows={}
    for line in (args.run_dir/'detections.jsonl').open():
        row=json.loads(line);key=round(float(row['t']),6)
        if key in rows:raise ValueError('Duplicate measurement timestamp')
        rows[key]=row
    results=[];track_labels={}
    for label in labels:
        if label['kind']!='identity':continue
        team=label.get('team','N/A');number=label['label']
        if team not in ('Arsenal','Liverpool','N/A',''):raise ValueError('Identity team must be explicitly selected, not inferred from notes')
        valid_number=number.isdigit() and 1<=int(number)<=99
        link=associate(label,rows.get(round(label['video_s'],6)),point_kind=args.point_kind)
        result={**label,**link,'label_complete':valid_number and team in ('Arsenal','Liverpool')}
        results.append(result)
        if link['track'] is not None and result['label_complete']:
            track_labels.setdefault(link['track'],set()).add((team,int(number)))
    conflicts={str(track):sorted(values) for track,values in track_labels.items() if len(values)>1}
    conditions=('Independent human shirt/team clicks linked only to nearby current-image feet at exact timestamps. '
        '20-pixel distance and six-pixel ambiguity margin are review-association rules, not model confidence. '
        'N/A labels, missed detections and ambiguous feet are not forced to identities. Conflicting labels on '
        'one anonymous track may indicate an ID switch or annotation mistake and require review. '
        'No pitch, hidden-player, automatic jersey recognition or named-roster accuracy is established. '
        'Labels are evaluation-only and never used to train or rewrite a match.')
    if args.basis=='assistant_provisional':
        conditions=conditions.replace('Independent human shirt/team clicks',
            'Provisional assistant shirt/team clicks, not independently verified human ground truth,')
    if args.point_kind=='body':
        conditions=f'{args.basis} shirt/team body clicks associated only with a unique containing person box at the exact current-main-camera timestamp. Overlapping boxes and missing/cut observations are not forced to identities. No clicked foot, pitch accuracy, automatic shirt recognition or hidden-player truth is established. Conflicting or incorrectly selected team labels require review. These labels never train or rewrite a match.'
    output=dict(annotation_basis=args.basis,point_kind=args.point_kind,conditions=conditions,rows=results,conflicting_track_labels=conflicts,
                csv_sha256=hashlib.sha256(args.csv.read_bytes()).hexdigest(),run_dir=str(args.run_dir),
                source_detections_sha256=hashlib.sha256((args.run_dir/'detections.jsonl').read_bytes()).hexdigest())
    (args.root/f'{args.name}.json').write_text(json.dumps(output,indent=2))
    display=[[r['frame_id'],r.get('team') or 'N/A',r['label'],r['status'],r['track'] if r['track'] is not None else 'N/A',
              round(r['distance_pixels'],1) if 'distance_pixels' in r else 'N/A',r.get('reason',''),r.get('notes','')] for r in results]
    conflict_rows=[[track,', '.join(f'{team} #{number}' for team,number in values)] for track,values in conflicts.items()]
    (args.root/f'{args.name}.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<title>Private human identity review</title><style>body{font:16px system-ui;margin:30px;max-width:1200px}'
        'table{border-collapse:collapse;width:100%}td,th{border:1px solid #aaa;padding:8px}</style>'
        '<h1>Human shirt labels and anonymous tracks</h1><p>'+escape(conditions)+'</p><p>Annotation basis: '
        +escape(args.basis)+'</p>'+table(['Frame','Human team','Human shirt','Association','Anonymous ID',
            'Foot distance px','Reason','Note'],display)+'<h2>Labels requiring track review</h2>'
        +table(['Anonymous ID','Different human labels'],conflict_rows)+'</html>')
    print(args.root/f'{args.name}.html',flush=True)


if __name__=='__main__':main()
