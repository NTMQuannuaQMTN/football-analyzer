"""Refine an existing pitch calibration against paint in the current image.

No player truth or future frame is read. Alignment is an image diagnostic,
not a calibrated guarantee of pitch-coordinate accuracy.
"""
import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares


def pitch_markings(step=.5):
    points=[]
    def line(a,b):
        n=max(2,int(np.linalg.norm(np.asarray(b)-a)/step)+1)
        points.extend(np.linspace(a,b,n))
    for a,b in [((0,0),(105,0)),((0,68),(105,68)),((0,0),(0,68)),((105,0),(105,68)),((52.5,0),(52.5,68))]:
        line(a,b)
    for depth,top,bottom in ((16.5,13.84,54.16),(5.5,24.84,43.16)):
        for start,x in ((0,depth),(105,105-depth)):
            line((start,top),(x,top));line((x,top),(x,bottom));line((x,bottom),(start,bottom))
    angle=np.linspace(0,2*np.pi,180)
    points.extend(np.column_stack([52.5+9.15*np.cos(angle),34+9.15*np.sin(angle)]))
    arc=np.linspace(-np.arccos(5.5/9.15),np.arccos(5.5/9.15),50)
    for spot,sign in ((11,1),(94,-1)):
        points.extend(np.column_stack([spot+sign*9.15*np.cos(arc),34+9.15*np.sin(arc)]))
    return np.asarray(points)


def project(matrix,xy):
    hp=np.column_stack([xy,np.ones(len(xy))])@matrix.T
    return hp[:,:2]/hp[:,2:]


def refine(image,image_to_pitch,boxes=()):
    height,width=image.shape[:2]
    small=cv2.resize(image,(640,360))
    hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
    grass=((hsv[:,:,0]>30)&(hsv[:,:,0]<95)&(hsv[:,:,1]>45)&(hsv[:,:,2]>35)).astype(np.uint8)
    grass=cv2.morphologyEx(grass,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    n,components,stats,_=cv2.connectedComponentsWithStats(grass)
    if n>1:
        grass=(components==1+np.argmax(stats[1:,cv2.CC_STAT_AREA])).astype(np.uint8)
    near_grass=cv2.dilate(grass,np.ones((11,11),np.uint8))>0
    value=hsv[:,:,2].astype(float)
    contrast=value-cv2.GaussianBlur(value,(0,0),4)
    # Bright desaturated grass must not count as white paint. Retain narrow,
    # locally bright strokes, including lines in stadium shadows.
    paint=(near_grass&(hsv[:,:,1]<60)&(value>75)&(contrast>10)).astype(np.uint8)
    for x1,y1,x2,y2 in boxes:
        x1,x2=np.clip(np.asarray([x1,x2])*640/width,0,639).astype(int)
        y1,y2=np.clip(np.asarray([y1,y2])*360/height,0,359).astype(int)
        paint[y1:y2+1,x1:x2+1]=0
    diagnostic=dict(accepted=False,reason="insufficient pitch paint",paint_pixels=int(paint.sum()),alignment_basis="current image paint; not position ground truth")
    if paint.sum()<100:
        return image_to_pitch,diagnostic
    distance=cv2.distanceTransform(1-paint,cv2.DIST_L2,5).astype(float)
    world=pitch_markings()
    pixel=project(np.linalg.inv(image_to_pitch),world)
    norm=pixel/[width,height]
    keep=np.isfinite(norm).all(1)&(norm[:,0]>.02)&(norm[:,0]<.98)&(norm[:,1]>.02)&(norm[:,1]<.98)
    norm=norm[keep]
    diagnostic["visible_line_samples"]=len(norm)
    if len(norm)<40:
        return image_to_pitch,diagnostic
    def matrix(p):
        return np.array([[1+p[0],p[1],p[2]],[p[3],1+p[4],p[5]],[p[6],p[7],1.]])
    def distances(p):
        xy=project(matrix(p),norm)*[640,360]
        d=map_coordinates(distance,[xy[:,1],xy[:,0]],order=1,mode="constant",cval=30)
        return np.minimum(d,30)
    def residual(p):
        return np.concatenate([distances(p)/3,p*80])
    before=distances(np.zeros(8))
    fit=least_squares(residual,np.zeros(8),bounds=(-.12,.12),loss="soft_l1",f_scale=1,max_nfev=60)
    after=distances(fit.x)
    diagnostic.update(before_median_px=float(np.median(before)*width/640),
                      after_median_px=float(np.median(after)*width/640),
                      before_mean_px=float(before.mean()*width/640),after_mean_px=float(after.mean()*width/640))
    if after.mean()>=before.mean()*.85 or np.median(after)>3 or np.max(np.abs(fit.x))>.119:
        diagnostic["reason"]="alignment gain insufficient or correction reached safety bound"
        return image_to_pitch,diagnostic
    delta=np.diag([width,height,1])@matrix(fit.x)@np.diag([1/width,1/height,1])
    result=image_to_pitch@np.linalg.inv(delta)
    diagnostic.update(accepted=True,reason="paint alignment improved; pitch accuracy requires independent evaluation")
    return result,diagnostic
