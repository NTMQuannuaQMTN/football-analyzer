"""Score existing cameras against independent, human-marked pitch landmarks.

Never refits a camera or uses these labels for normal inference. Shirt labels
are inventoried separately; they are not full tracking ground truth.
"""
import argparse
import csv
import hashlib
from html import escape
import json
from pathlib import Path

import numpy as np

from cv.soccernet import Calibration,DEFAULT_GAME,on_main_camera


def cached_camera(run_dir):
    run_dir=Path(run_dir);meta=json.loads((run_dir/'run.json').read_text())
    if meta.get('camera_H_coordinates')!='normalised_0_to_1_image_to_pitch_metres':
        raise ValueError('Candidate must declare normalised-image camera matrix coordinates')
    matrices={}
    for line in (run_dir/'detections.jsonl').open():
        row=json.loads(line);key=round(float(row['t']),6)
        if key in matrices:raise ValueError('Duplicate candidate camera timestamp')
        h=row.get('H')
        matrix=None if h is None else np.asarray(h,float)
        if matrix is not None and (matrix.shape!=(3,3) or not np.isfinite(matrix).all()):
            raise ValueError('Invalid candidate camera matrix')
        matrices[key]=None if matrix is None else matrix@np.diag([1/1280.,1/720.,1.])
    return lambda t:matrices.get(round(float(t),6))


def read_labels(path,frames,basis='human'):
    if basis not in ('human','assistant_provisional'):raise ValueError('Unknown annotation basis')
    labels=[];known={f['id']:f for f in frames}
    with Path(path).open(newline='') as stream:
        for number,r in enumerate(csv.DictReader(stream),2):
            if 'UI SMOKE TEST ONLY' in r.get('notes',''):
                raise ValueError(f'Row {number}: fabricated UI smoke clicks are not evaluation references')
            notes=r.get('notes','').lower()
            if basis=='human' and ('assistant visual reading' in notes or 'not independent human gt' in notes):
                raise ValueError(f'Row {number}: assistant-provisional readings require --basis assistant_provisional; they are not human ground truth')
            if r.get('frame_id') not in known:raise ValueError(f'Row {number}: unknown frame')
            f=known[r['frame_id']]
            values=[float(r[k]) for k in ('video_s','pixel_x','pixel_y')]
            if not np.isfinite(values).all() or abs(values[0]-f['video_s'])>.001:raise ValueError(f'Row {number}: timestamp does not match frame')
            if not 0<=values[1]<=1280 or not 0<=values[2]<=720:raise ValueError(f'Row {number}: image point is outside frame')
            if r['kind'] not in ('landmark','identity','unclear'):raise ValueError(f'Row {number}: unsupported label kind')
            r.update(video_s=values[0],pixel_x=values[1],pixel_y=values[2])
            if r['kind']=='landmark':
                world=np.array([float(r['world_x']),float(r['world_y'])])
                if not np.isfinite(world).all() or not 0<=world[0]<=105 or not 0<=world[1]<=68:raise ValueError(f'Row {number}: invalid pitch landmark')
                r.update(world_x=float(world[0]),world_y=float(world[1]))
            labels.append(r)
    return labels


def score(labels,camera,eligible):
    errors=[];rows=[];identity=[]
    for label in labels:
        if label['kind']=='identity':identity.append(label);continue
        if label['kind']!='landmark':continue
        row=dict(label,status='N/A',error_m=None)
        if not eligible(label['video_s']):row['reason']='Replay or close-up; not current-time camera'
        else:
            h=camera(label['video_s'])
            if h is None:row['reason']='Camera estimate missing'
            else:
                p=h@np.array([label['pixel_x'],label['pixel_y'],1.])
                if abs(p[2])<1e-10:row['reason']='Projection is at horizon'
                else:
                    point=p[:2]/p[2]
                    if not np.isfinite(point).all():row['reason']='Nonfinite camera projection'
                    else:
                        e=float(np.linalg.norm(point-np.array([label['world_x'],label['world_y']])))
                        row.update(status='scored',predicted_x=float(point[0]),predicted_y=float(point[1]),error_m=e);errors.append(e)
        rows.append(row)
    return dict(landmark_labels=len(rows),scored_landmarks=len(errors),readable_identity_labels=len(identity),
        mean_landmark_error_m=float(np.mean(errors)) if errors else None,median_landmark_error_m=float(np.median(errors)) if errors else None,
        p90_landmark_error_m=float(np.percentile(errors,90)) if errors else None,rows=rows,identity_labels=identity,
        conditions='Independent manual pitch landmarks scored against unchanged CCBV estimates. No camera fit, hidden-player reference, named tracking score or training input. Human clicks and coordinate orientation need review. This is landmark accuracy, not player reconstruction accuracy.')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--csv',required=True);ap.add_argument('--root',default='cv/out/audit8_20261011/manual_review')
    ap.add_argument('--offset',type=float,default=0.,help='Explicit experimental camera timestamp offset, seconds')
    ap.add_argument('--name',default='manual_camera_scores',help='Private output basename')
    ap.add_argument('--basis',choices=['human','assistant_provisional'],default='human',help='Do not describe unverified assistant readings as independent human ground truth')
    ap.add_argument('--camera-run',help='Score a private camera-only candidate at exact recorded frame timestamps; no refit')
    a=ap.parse_args()
    if Path(a.name).name!=a.name:raise ValueError('Output name must be a basename')
    root=Path(a.root);frames=json.loads((root/'frames.json').read_text());labels=read_labels(a.csv,frames,a.basis)
    if a.camera_run:
        if a.offset!=0:raise ValueError('Candidate cameras have exact frame timestamps; do not also apply a clock offset')
        projection=cached_camera(a.camera_run)
    else:
        camera=Calibration(DEFAULT_GAME,1);projection=lambda t:camera.at(t+a.offset)
    result=score(labels,projection,lambda t:on_main_camera(DEFAULT_GAME,1,t))
    result['camera_run']=a.camera_run
    if a.camera_run:result['conditions']='Independent manual landmarks scored against a frozen private candidate at exact recorded timestamps. Camera coordinate units are checked. No fitting, hidden-player truth or training input. Human clicks and orientation need review; this is landmark accuracy, not full-player reconstruction accuracy.'
    result['camera_offset_s']=a.offset
    result['annotation_basis']=a.basis
    if a.basis=='assistant_provisional':
        result['conditions']='Provisional assistant visual landmark readings; not independently verified human ground truth. Review the source image and CSV notes before relying on these errors. No camera fit, hidden-player reference or training input.'
    result['csv_sha256']=hashlib.sha256(Path(a.csv).read_bytes()).hexdigest()
    (root/f'{a.name}.json').write_text(json.dumps(result,indent=2))
    rows=''.join('<tr>'+''.join(f'<td>{escape(str(v))}</td>' for v in [r['frame_id'],r['label'],r['status'],f"{r['error_m']:.2f} m" if r['error_m'] is not None else 'N/A',r.get('reason',''),r.get('notes','')])+'</tr>' for r in result['rows'])
    mean=f"{result['mean_landmark_error_m']:.2f} m" if result['mean_landmark_error_m'] is not None else 'N/A'
    html=f'''<!doctype html><meta charset="utf-8"><title>Private manual camera evaluation</title><style>body{{font:16px system-ui;margin:30px;max-width:1000px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #aaa;padding:8px}}</style><h1>Private manual camera evaluation</h1><p>{escape(result['conditions'])}</p><p>Scored landmarks: {len(result['rows'])} labelled, {result['scored_landmarks']} scored. Mean landmark error: {mean}.</p><table><thead><tr><th>Frame</th><th>Landmark</th><th>Status</th><th>Error</th><th>Reason</th><th>Annotation note</th></tr></thead><tbody>{rows}</tbody></table>'''
    (root/f'{a.name}.html').write_text(html);print(root/f'{a.name}.html')


if __name__=='__main__':main()
