"""Image-only physical plausibility checks, not calibrated position probability."""
import cv2
import numpy as np


def assess_camera(image, normalised_image_to_pitch):
    """Use a canonical 1920x1080 pixel grid for cross-resolution scale checks.

    The supplied image may be resized. These gradients are not necessarily
    metres per native source-video pixel (Arsenal's local video is 1280x720).
    """
    if normalised_image_to_pitch is None:return dict(status='unsupported',reason='no constrained camera')
    small=cv2.resize(image,(256,144));hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
    mask=((hsv[:,:,0]>30)&(hsv[:,:,0]<95)&(hsv[:,:,1]>30)&(hsv[:,:,2]>25)).astype(np.uint8)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    count,components,stats,_=cv2.connectedComponentsWithStats(mask)
    if count<=1:return dict(status='unsupported',reason='no identifiable grass region')
    label=1+np.argmax(stats[1:,cv2.CC_STAT_AREA]);mask=components==label
    y,x=np.where(mask);take=np.linspace(0,len(x)-1,min(len(x),500),dtype=int)
    if len(take)<30:return dict(status='unsupported',reason='insufficient grass samples')
    xy=np.column_stack([x[take]/255,y[take]/143]);h=np.asarray(normalised_image_to_pitch,float)
    hp=np.column_stack([xy,np.ones(len(xy))])@h.T;denominator=hp[:,2]
    with np.errstate(divide='ignore',invalid='ignore'):
        world=hp[:,:2]/denominator[:,None]
    finite=np.isfinite(world).all(1)
    within=finite&(world[:,0]>=-3)&(world[:,0]<=108)&(world[:,1]>=-3)&(world[:,1]<=71)
    signs=np.sign(denominator);pole_fraction=float(min(np.mean(signs>0),np.mean(signs<0)))
    gradients=[]
    for step in ([1/1920,0],[0,1/1080]):
        projected=np.column_stack([xy+step,np.ones(len(xy))])@h.T
        with np.errstate(divide='ignore',invalid='ignore'):
            shifted=projected[:,:2]/projected[:,2:]
        gradients.append(np.linalg.norm(shifted-world,axis=1))
    scale=np.maximum(*gradients);scale=scale[np.isfinite(scale)]
    if not len(scale):return dict(status='unsupported',reason='nonfinite projection')
    return dict(status='measured',grass_samples=len(xy),grass_inside_pitch_fraction=float(within.mean()),
                horizon_crossing_fraction=pole_fraction,median_metres_per_original_pixel=float(np.median(scale)),
                p90_metres_per_original_pixel=float(np.percentile(scale,90)),
                median_metres_per_reference_pixel=float(np.median(scale)),
                p90_metres_per_reference_pixel=float(np.percentile(scale,90)),
                jacobian_reference_image_size=[1920,1080],
                legacy_scale_key_note='original_pixel keys retained for compatibility; values use the canonical reference grid, not arbitrary native video resolution',
                basis='Current-image grass and projective Jacobian on a canonical 1920x1080 reference grid; plausibility only. A wrong orientation or semantic correspondence can pass. Not position confidence.')


def plausible(diagnostic,inside=.75,scale=.5):
    return diagnostic.get('status')=='measured' and diagnostic['grass_inside_pitch_fraction']>=inside and diagnostic['horizon_crossing_fraction']<=.01 and diagnostic['p90_metres_per_original_pixel']<=scale
