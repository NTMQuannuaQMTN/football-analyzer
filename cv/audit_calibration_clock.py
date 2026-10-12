"""Image-paint camera/video clock diagnosis, not player-position ground truth."""
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

from cv.refine_calibration import pitch_markings
from cv.soccernet import Calibration,DEFAULT_GAME


def paint_distance(image,boxes=()):
    small=cv2.resize(image,(640,360));hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
    grass=((hsv[:,:,0]>30)&(hsv[:,:,0]<95)&(hsv[:,:,1]>45)&(hsv[:,:,2]>35)).astype(np.uint8)
    grass=cv2.morphologyEx(grass,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    n,components,stats,_=cv2.connectedComponentsWithStats(grass)
    if n>1:grass=(components==1+np.argmax(stats[1:,cv2.CC_STAT_AREA])).astype(np.uint8)
    value=hsv[:,:,2].astype(float);contrast=value-cv2.GaussianBlur(value,(0,0),4)
    paint=((cv2.dilate(grass,np.ones((11,11),np.uint8))>0)&(hsv[:,:,1]<60)&(value>75)&(contrast>10)).astype(np.uint8)
    for box in boxes:
        x1,y1,x2,y2=np.array(box)*[640/image.shape[1],360/image.shape[0],640/image.shape[1],360/image.shape[0]]
        x1,x2=np.clip([x1,x2],0,639).astype(int);y1,y2=np.clip([y1,y2],0,359).astype(int);paint[y1:y2+1,x1:x2+1]=0
    return cv2.distanceTransform(1-paint,cv2.DIST_L2,5).astype(float),int(paint.sum())


def alignment(distance,h,width=1280,height=720):
    if h is None:return None
    inverse=np.linalg.inv(h);world=pitch_markings(step=.5)
    ref=h@np.array([width/2,height-1,1.]);ref/=ref[2]
    if (inverse@ref)[2]<0:inverse=-inverse
    hp=np.column_stack([world,np.ones(len(world))])@inverse.T
    with np.errstate(divide='ignore',invalid='ignore'):pixel=hp[:,:2]/hp[:,2:]
    norm=pixel/[width,height]
    keep=np.isfinite(norm).all(1)&(hp[:,2]>0)&(norm[:,0]>.02)&(norm[:,0]<.98)&(norm[:,1]>.02)&(norm[:,1]<.98)
    norm=norm[keep]
    if len(norm)<40:return None
    d=map_coordinates(distance,[norm[:,1]*360,norm[:,0]*640],order=1,mode='constant',cval=30)
    return dict(samples=len(norm),mean_px_720=float(np.minimum(d,30).mean()*width/640),median_px_720=float(np.median(d)*width/640))


def main():
    root=Path('cv/out/audit8_20261011');frames=json.loads((root/'broadcast/frames.json').read_text());cal=Calibration(DEFAULT_GAME,1)
    old={round(r['t'],2):r for r in map(json.loads,Path('cv/out/training_20261011/arsenal_corrected_mps/detections.jsonl').read_text().splitlines())}
    offsets=np.arange(-2,2.01,.25).tolist();rows=[]
    for part,low,high in [('selection',1440,1740),('check',1740,2041)]:
        available=[r for r in frames if low<=r['t']<high and r['image'] is not None]
        indexes=np.linspace(0,len(available)-1,40,dtype=int)
        for i in indexes:
            row=available[i];image=cv2.imread(str(root/'broadcast/images'/row['image']))
            boxes=[d['box'] for d in old.get(round(row['t'],2),{}).get('dets',[]) if d['cls']=='person']
            distance,count=paint_distance(image,boxes)
            if count<100:continue
            scores={str(o):alignment(distance,cal.at(row['t']+o)) for o in offsets}
            if not all(v is not None for v in scores.values()):continue
            rows.append(dict(part=part,t=row['t'],image=row['image'],paint_pixels=count,scores=scores))
    summary={}
    for part in ['selection','check']:
        chosen=[r for r in rows if r['part']==part]
        summary[part]={str(o):dict(n=len(chosen),mean_alignment_px_720=float(np.mean([r['scores'][str(o)]['mean_px_720'] for r in chosen])) if chosen else None) for o in offsets}
    eligible=summary['selection'];selected=min(eligible,key=lambda k:eligible[k]['mean_alignment_px_720']) if rows and all(v['mean_alignment_px_720'] is not None for v in eligible.values()) else None
    result=dict(summary=summary,selected_offset_s=float(selected) if selected is not None else None,rows=rows,
        conditions='Same eligible image frames for every offset; first clip-half selects, later half checks paint agreement only. CCBV metadata names 1_HQ_25.mkv while the local source is 1_720p.mkv. A clock offset or better paint score does not establish ground-truth player accuracy; no export is changed. Existing tracked boxes mask shirts, no manual labels.')
    (root/'calibration_clock.json').write_text(json.dumps(result,indent=2));print('selected',selected,'summary',summary,flush=True)


if __name__=='__main__':main()
