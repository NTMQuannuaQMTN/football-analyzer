"""Causal camera propagation from current/past image ground-plane features."""
import cv2
import numpy as np


def ground_mask(image):
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    grass=((hsv[:,:,0]>30)&(hsv[:,:,0]<95)&(hsv[:,:,1]>35)&(hsv[:,:,2]>30)).astype(np.uint8)
    grass=cv2.morphologyEx(grass,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    white=((hsv[:,:,1]<85)&(hsv[:,:,2]>120)).astype(np.uint8)
    return cv2.dilate(white,np.ones((5,5),np.uint8))*grass*255


def ground_motion(previous,current):
    """Current normalised image -> previous normalised image, or abstain.

    White markings near grass reduce moving-player/advertising features.
    Forward/backward flow, RANSAC and spatial extent reject weak solutions.
    This is a quality gate, not a calibrated probability.
    """
    size=(640,360);a=cv2.resize(previous,size);b=cv2.resize(current,size)
    ga=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY);gb=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)
    p=cv2.goodFeaturesToTrack(ga,400,.01,8,mask=ground_mask(a))
    if p is None or len(p)<16:return None,dict(status='unsupported',reason='few pitch features')
    q,ok,_=cv2.calcOpticalFlowPyrLK(ga,gb,p,None,winSize=(25,25),maxLevel=3)
    if q is None:return None,dict(status='unsupported',reason='flow failed')
    back,reverse,_=cv2.calcOpticalFlowPyrLK(gb,ga,q,None,winSize=(25,25),maxLevel=3)
    if back is None:return None,dict(status='unsupported',reason='reverse flow failed')
    keep=(ok.ravel()>0)&(reverse.ravel()>0)&(np.linalg.norm(back[:,0]-p[:,0],axis=1)<1)
    p=p[keep,0];q=q[keep,0]
    if len(p)<16:return None,dict(status='unsupported',reason='few consistent features')
    cv2.setRNGSeed(11026);h,inliers=cv2.findHomography(q,p,cv2.RANSAC,2)
    if h is None or inliers is None:return None,dict(status='unsupported',reason='no ground homography')
    mask=inliers.ravel().astype(bool);count=int(mask.sum());fraction=float(mask.mean())
    extent=cv2.contourArea(cv2.convexHull(q[mask].astype(np.float32)))/(640*360) if count>=3 else 0
    if count<16 or fraction<.6 or extent<.015:return None,dict(status='unsupported',reason='weak spatial support',inliers=count,inlier_fraction=fraction,extent=float(extent))
    transform=np.diag([640.,360.,1.]);normal=np.linalg.inv(transform)@h@transform
    corners=np.array([[.05,.05,1],[.95,.05,1],[.95,.95,1],[.05,.95,1]])@normal.T
    denominator=corners[:,2]
    if not np.isfinite(normal).all() or (denominator>0).any() and (denominator<0).any():return None,dict(status='unsupported',reason='projective pole')
    return normal,dict(status='measured',inliers=count,inlier_fraction=fraction,extent=float(extent),basis='Current/past white ground features; no player labels')


def propagate(previous_image_to_pitch,current_to_previous):
    if previous_image_to_pitch is None or current_to_previous is None:return None
    h=np.asarray(previous_image_to_pitch)@np.asarray(current_to_previous)
    return h/np.linalg.norm(h) if np.isfinite(h).all() else None
