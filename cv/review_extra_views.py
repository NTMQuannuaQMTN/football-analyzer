"""Private close-up/goal image observations; never uses replay pictures as current match positions."""
import argparse,json,hashlib
from pathlib import Path
import cv2,numpy as np,torch
from ultralytics import YOLO
from cv.soccernet import DEFAULT_GAME,game_dir,camera_segments
from cv.broadcast_views import view_kind
from cv.audit_closeup_retrain import predict
from cv.train_camera_audit import ResNetPitch
from cv.camera_quality import assess_camera,plausible
from cv.gsr import project
from cv.session_report import table

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);cv2.setNumThreads(1);torch.manual_seed(11026);cv2.setRNGSeed(11026)
    detector_path=Path('cv/weights/yolo_gsr.pt');camera_path=Path('cv/out/apply_train7h_20261012_053349/epl_trained/camera.pt')
    generic_path=Path('cv/weights/yolo11m.pt')
    if not generic_path.is_file():raise FileNotFoundError('Local general detector missing; no download attempted')
    detector=YOLO(detector_path);generic=YOLO(generic_path);camera=ResNetPitch();camera.load_state_dict(torch.load(camera_path,weights_only=False)['state']);camera.eval()
    cap=cv2.VideoCapture(str(game_dir(DEFAULT_GAME)/'1_720p.mkv'));segments=camera_segments(DEFAULT_GAME,1)
    times=[1669.,1725.,1734.,1742.,1747.,1810.,1811.68,1817.,1840.,1850.];records=[];html=[]
    for t in times:
        shot=next((s for start,end,s in segments if start<=t<end),None)
        cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,image=cap.read()
        if not ok:raise RuntimeError(f'No frame at {t}')
        image=cv2.resize(image,(1280,720));original=a.out/f'{t:g}_source.jpg';cv2.imwrite(str(original),image)
        realtime=shot is not None and shot.get('replay','real-time')=='real-time'
        h=predict(camera,original,'cpu') if realtime else None
        supported=h is not None and plausible(assess_camera(image,h),.75,.25)
        active_detector=generic if shot and view_kind(shot)=='closeup' else detector
        prediction=active_detector.predict(image,imgsz=1280,conf=.15,agnostic_nms=True,device='cpu',verbose=False)[0]
        feet=[];rows=[]
        for box,cls,confidence in zip(prediction.boxes.xyxy.cpu().numpy(),prediction.boxes.cls.cpu().numpy(),prediction.boxes.conf.cpu().numpy()):
            if active_detector.names[int(cls)] not in ('person','player','goalkeeper','referee'):continue
            x=(box[0]+box[2])/2;y=box[3];world=project(h,np.array([[x/1280,y/720]]))[0] if supported else None
            clipped=box[3]>=image.shape[0]-2
            valid=not clipped and world is not None and np.isfinite(world).all() and -3<=world[0]<=108 and -3<=world[1]<=71
            foot=dict(image_xy=[float(x),float(y)],pitch_xy=world.tolist() if valid else None,detector_confidence=float(confidence),identity='anonymous',clipped=bool(clipped),point_basis='Box bottom proxy; cropped bodies do not establish foot position',view=shot['label'] if shot else 'N/A');feet.append(foot)
            n=len(feet);cv2.rectangle(image,tuple(box[:2].astype(int)),tuple(box[2:].astype(int)),(50,220,240),1);cv2.circle(image,(int(x),int(y)),4,(50,220,240),-1);cv2.putText(image,str(n),(int(x),int(y)-8),cv2.FONT_HERSHEY_SIMPLEX,.5,(50,220,240),1)
            rows.append([n,round(x,1),round(y,1),round(float(confidence),3),', '.join(f'{v:.1f}' for v in world) if valid else 'N/A'])
        name=f'{t:g}_detections.jpg';cv2.imwrite(str(a.out/name),image)
        kind=view_kind(shot) if shot else 'N/A';basis='Image-only experimental camera; unverified metre positions' if supported else 'Image pixels only; pitch calibration N/A'
        if not realtime:basis='Broadcast replay: original match time N/A; excluded from match inference'
        records.append(dict(video_s=t,clip_s=t-1440,kind=kind,realtime=realtime,pitch_camera_supported=bool(supported),basis=basis,feet=feet))
        html.append(f'<h2>Video {t:g}s / clip {t-1440:g}s · {kind}</h2><p>{basis}. Anonymous observations include officials/crowd; no shirt identities assigned. Box bottoms are foot proxies; cropped bodies cannot establish pitch coordinates.</p><img width="960" src="{name}" alt="Private image-space detection proxies" />'+table(['Detection','Box-bottom x px','Box-bottom y px','Detector score','Pitch x,y m'],rows))
        print(t,kind,'people',len(feet),'pitch supported',supported,flush=True)
    cap.release();(a.out/'observations.json').write_text(json.dumps(records,indent=2))
    (a.out/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Private close-up and goal review</title><style>body{font:16px system-ui;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:8px}</style><h1>Private extra-view observations</h1><p>Local permitted asset only; do not distribute. Pixels can locate visible feet inside a close-up without locating them on the global pitch. Detector scores are not position confidence. Replay time is not current match time.</p>'+''.join(html))
    (a.out/'provenance.json').write_text(json.dumps(dict(detector_sha256=hashlib.sha256(detector_path.read_bytes()).hexdigest(),closeup_detector_sha256=hashlib.sha256(generic_path.read_bytes()).hexdigest(),camera_sha256=hashlib.sha256(camera_path.read_bytes()).hexdigest(),source_game=DEFAULT_GAME,times=times,seed=11026,training=False,promoted=False),indent=2))
if __name__=='__main__':main()
