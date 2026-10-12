"""Local-only video/anonymous-measurement review; never served by the app."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from cv.soccernet import game_dir


def render(run_dir,output,start,duration,fps,show_unconfirmed_pitch=False):
    run=json.loads((run_dir/'run.json').read_text())
    if show_unconfirmed_pitch and run.get('camera_H_coordinates')!='normalised_0_to_1_image_to_pitch_metres':
        raise ValueError('Unconfirmed pitch review requires explicitly normalised camera matrices')
    rows=[json.loads(line) for line in (run_dir/'detections.jsonl').open()]
    times=np.array([r['t'] for r in rows])
    if not (np.diff(times)>0).all():raise ValueError('Measurements need increasing timestamps')
    video=game_dir(run['game'])/f"{run['half']}_720p.mkv"
    cap=cv2.VideoCapture(str(video));source_fps=cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or not source_fps:raise FileNotFoundError(video)
    output.parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1440,620))
    if not writer.isOpened():raise RuntimeError('Private movie writer unavailable')
    video_start=run['start']+start
    cap.set(cv2.CAP_PROP_POS_MSEC,video_start*1000);index=-1
    def point(x,y):return int(984+x*4.1),int(154+y*4.1)
    reviewed=[]
    try:
        for frame in range(round(duration*fps)):
            t=video_start+frame/fps;target=round(frame*source_fps/fps)
            while index<target:
                ok,image=cap.read();index+=1
                if not ok:raise RuntimeError(f'Video unreadable at {t}')
            i=int(np.clip(np.searchsorted(times,t,side='right')-1,0,len(rows)-1));row=rows[i]
            stale=t-row['t']>.251
            canvas=np.full((620,1440,3),24,np.uint8);canvas[70:610,:960]=cv2.resize(image,(960,540))
            cv2.rectangle(canvas,point(0,0),point(105,68),(40,90,45),-1)
            cv2.rectangle(canvas,point(0,0),point(105,68),(175,195,175),1)
            cv2.line(canvas,point(52.5,0),point(52.5,68),(175,195,175),1);cv2.circle(canvas,point(52.5,34),round(9.15*4.1),(175,195,175),1)
            for x in [0,88.5]:cv2.rectangle(canvas,point(x,13.84),point(x+16.5,54.16),(175,195,175),1)
            measured=0;extra=0
            if row['live'] and not stale:
                for d in row['dets']:
                    if d['cls']!='person':continue
                    box=d.get('raw_box',d['box']);a=(round(box[0]*.75),round(box[1]*.75)+70);b=(round(box[2]*.75),round(box[3]*.75)+70)
                    cv2.rectangle(canvas,a,b,(240,240,240),1)
                    text=f"T{d['id']}";cv2.putText(canvas,text,(a[0],max(82,a[1]-3)),cv2.FONT_HERSHEY_SIMPLEX,.32,(255,255,255),1)
                    foot=(round((box[0]+box[2])*.375),round(box[3]*.75)+70);cv2.drawMarker(canvas,foot,(0,220,255),cv2.MARKER_CROSS,9,1)
                    p=point(d['px'],d['py'])
                    cv2.circle(canvas,p,4,(240,240,240),-1);cv2.putText(canvas,text,(p[0]+4,p[1]-4),cv2.FONT_HERSHEY_SIMPLEX,.3,(255,255,255),1)
                    measured+=1
                for d in row.get('unassigned_dets',[]):
                    if d.get('cls') not in ('person',None):continue
                    b=d['box'];cv2.rectangle(canvas,(round(b[0]*.75),round(b[1]*.75)+70),(round(b[2]*.75),round(b[3]*.75)+70),(0,180,210),1)
                    if show_unconfirmed_pitch and d.get('cls')=='person' and row.get('H') is not None:
                        h=np.asarray(row['H'],float);foot=np.array([(b[0]+b[2])/2/1280,b[3]/720,1.])
                        hp=h@foot
                        if abs(hp[2])<1e-10:continue
                        world=hp[:2]/hp[2]
                        if not np.isfinite(world).all() or not (-3<world[0]<108 and -3<world[1]<71):continue
                        extra+=1;p=point(*world)
                        cv2.drawMarker(canvas,p,(0,180,210),cv2.MARKER_DIAMOND,8,1)
                        cv2.putText(canvas,f'U{extra}',(p[0]+4,p[1]-4),cv2.FONT_HERSHEY_SIMPLEX,.3,(0,180,210),1)
            cv2.putText(canvas,f'PRIVATE | video {t:.1f}s | measurement {row["t"]:.1f}s | anonymous tracks only',(12,28),cv2.FONT_HERSHEY_SIMPLEX,.6,(245,245,245),1)
            status=f'{measured} projected measurements; camera and identities unverified' if row['live'] and not stale else 'Replay / unsupported camera: no current-time pitch measurements'
            if show_unconfirmed_pitch and row['live'] and not stale:status=f'{measured} anonymous tracks + {extra} unconfirmed detections projected; camera unverified'
            cv2.putText(canvas,status,(12,53),cv2.FONT_HERSHEY_SIMPLEX,.5,(100,190,255),1)
            cv2.putText(canvas,'White: tracker-confirmed anonymous',(978,500),cv2.FONT_HERSHEY_SIMPLEX,.38,(230,230,230),1)
            cv2.putText(canvas,'Amber: unconfirmed detection / no identity' if show_unconfirmed_pitch else 'Amber box: unassigned / unsupported',(978,525),cv2.FONT_HERSHEY_SIMPLEX,.32,(100,190,255),1)
            cv2.putText(canvas,'No named player or hidden GT claim',(978,550),cv2.FONT_HERSHEY_SIMPLEX,.38,(230,230,230),1)
            writer.write(canvas);reviewed.append(dict(video_s=t,measurement_s=row['t'],supported=row['live'],stale=stale,projected_people=measured,unconfirmed_pitch_detections=extra))
    finally:writer.release();cap.release()
    output.with_suffix('.json').write_text(json.dumps(dict(run_dir=str(run_dir),video=str(video),start=start,duration=duration,fps=fps,
        camera_basis=run.get('camera'),camera_conditions=run.get('camera_conditions'),
        projection='Stored anonymous measurements with the run camera provenance; no pitch-point clamping or projection outside the camera. Slightly out-of-pitch measurements remain drawn.',
        timing='Most recent 5Hz measurement, at most 0.25s old; no temporal point interpolation or future measurement selection.',
        unconfirmed_pitch_review=show_unconfirmed_pitch,
        unconfirmed_conditions='Additional image detections projected through the same unverified camera, with no name, track confirmation or roster claim. Can include officials/false positives; no hidden estimate, clamping or coordinate repair.',
        licensing='Existing private local asset; do not distribute or embed in the public app',frames=reviewed),indent=2))
    print(output,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--start',type=float,default=345);parser.add_argument('--duration',type=float,default=30);parser.add_argument('--fps',type=int,default=10)
    parser.add_argument('--show-unconfirmed-pitch',action='store_true',help='Private camera review only; show anonymous unconfirmed image detections without assigning roster identities')
    args=parser.parse_args()
    if args.fps<=0 or args.duration<=0 or args.start<0:raise ValueError('Invalid private review interval')
    render(args.run_dir,args.out,args.start,args.duration,args.fps,args.show_unconfirmed_pitch)
