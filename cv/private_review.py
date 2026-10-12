"""Render a local-only broadcast/pitch review movie. Never served by the app."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from cv.soccernet import game_dir, on_main_camera


def render(match, output, start, duration, fps=10):
    data = json.loads(Path(match).read_text())
    source = data['meta']['source']
    video = game_dir(source['game']) / f"{source['half']}_720p.mkv"
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    frames = data['frames']; times = np.array([f['t'] for f in frames])
    order = data['meta']['player_order']
    roster = {p['id']: p for p in data['players']}
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (1440, 620))
    if not writer.isOpened():
        raise RuntimeError(f'Cannot create {output}')
    cap.set(cv2.CAP_PROP_POS_MSEC, (source['start'] + start) * 1000)
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    read_index = -1
    def point(xy):
        # Equal metres per pixel: a stretched pitch distorts spatial review.
        return (int(982 + float(xy[0])*4.1), int(160 + float(xy[1])*4.1))
    try:
        for k in range(int(duration*fps)):
            t = start + k/fps
            target = int(round(k*actual_fps/fps))
            while read_index < target:
                ok, image = cap.read(); read_index += 1
                if not ok:
                    raise RuntimeError(f'Video ended at {source["start"]+t:.2f}s')
            i = int(np.clip(np.searchsorted(times, t, side='right')-1, 0, len(frames)-1))
            f = frames[i]; following = frames[min(i+1,len(frames)-1)]
            alpha = np.clip((t-f['t'])/max(.001,following['t']-f['t']),0,1)
            # Interpolation is presentation-only in this explicitly retrospective movie.
            xy = (1-alpha)*np.asarray(f['xy'],float) + alpha*np.asarray(following['xy'],float)
            canvas = np.full((620,1440,3),(24,24,24),np.uint8)
            canvas[60:600,:960] = cv2.resize(image,(960,540))
            cv2.rectangle(canvas,point([0,0]),point([105,68]),(35,92,43),-1)
            cv2.rectangle(canvas,point([0,0]),point([105,68]),(160,190,160),1)
            cv2.line(canvas,point([52.5,0]),point([52.5,68]),(160,190,160),1)
            cv2.circle(canvas,point([52.5,34]),int(9.15*4.1),(160,190,160),1)
            for x in (0,88.5):
                cv2.rectangle(canvas,point([x,13.84]),point([x+16.5,54.16]),(160,190,160),1)
            poly = f.get('cam')
            if poly:
                cv2.polylines(canvas,[np.array([point(p) for p in poly],np.int32)],True,(220,220,220),1)
            for j,pid in enumerate(order):
                if not np.isfinite(xy[j]).all():
                    continue
                colour = (75,65,220) if pid.startswith('A') else (200,150,80)
                p = point(xy[j]); observed = bool(f.get('obs',[False]*len(order))[j])
                cv2.circle(canvas,p,8,colour,-1 if observed else 1)
                cv2.putText(canvas,str(roster[pid]['shirt']), (p[0]+8,p[1]+4),cv2.FONT_HERSHEY_SIMPLEX,.32,colour,1)
            cv2.putText(canvas,f"PRIVATE REVIEW | video {source['start']+t:.1f}s | clip {t:.1f}s",(16,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(240,240,240),1)
            if not on_main_camera(source['game'],source['half'],source['start']+t):
                cv2.putText(canvas,'Replay/close-up: do not compare current-time positions to this image',(16,53),cv2.FONT_HERSHEY_SIMPLEX,.48,(100,180,255),1)
            cv2.putText(canvas,'Replay estimates; solid=measured, ring=inferred',(972,555),cv2.FONT_HERSHEY_SIMPLEX,.38,(230,230,230),1)
            cv2.putText(canvas,'Shirt identities role-assigned, NOT verified',(972,580),cv2.FONT_HERSHEY_SIMPLEX,.38,(230,230,230),1)
            writer.write(canvas)
    finally:
        writer.release(); cap.release()
    output.with_suffix('.json').write_text(json.dumps(dict(match=str(Path(match).resolve()),video=str(video),start=start,duration=duration,fps=fps,retrospective=True,interpolation='Display only; 1 Hz source positions',licensing='Private local review only; do not distribute'),indent=2))
    print(output)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--match',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--start',type=float,default=345); ap.add_argument('--duration',type=float,default=60)
    ap.add_argument('--fps',type=int,default=10)
    a=ap.parse_args(); render(a.match,a.out,a.start,a.duration,a.fps)
