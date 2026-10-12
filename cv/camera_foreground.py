"""Experimental ground motion excluding actual detected people, never labelled boxes."""
import cv2
import numpy as np

from cv.camera_motion import ground_mask


def detected_ground_mask(image,boxes,original_size):
    mask=ground_mask(image)
    width,height=original_size
    scale=np.array([image.shape[1]/width,image.shape[0]/height]*2)
    for box in np.asarray(boxes,float).reshape(-1,4):
        x1,y1,x2,y2=np.rint(box*scale).astype(int)
        cv2.rectangle(mask,(x1-3,y1-3),(x2+3,y2+3),0,-1)
    return mask


def motion_without_people(previous,current,previous_boxes,current_boxes):
    size=(640,360);a=cv2.resize(previous,size);b=cv2.resize(current,size)
    ga=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY);gb=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)
    masks=[detected_ground_mask(image,boxes,(original.shape[1],original.shape[0])) for image,boxes,original in [(a,previous_boxes,previous),(b,current_boxes,current)]]
    p=cv2.goodFeaturesToTrack(ga,400,.01,8,mask=masks[0])
    if p is None or len(p)<16:return None,dict(status='unsupported',reason='few unoccluded pitch features')
    q,ok,_=cv2.calcOpticalFlowPyrLK(ga,gb,p,None,winSize=(25,25),maxLevel=3)
    if q is None:return None,dict(status='unsupported',reason='flow failed')
    back,reverse,_=cv2.calcOpticalFlowPyrLK(gb,ga,q,None,winSize=(25,25),maxLevel=3)
    if back is None:return None,dict(status='unsupported',reason='reverse flow failed')
    pixel=np.rint(q[:,0]).astype(int);inside=(pixel[:,0]>=0)&(pixel[:,0]<640)&(pixel[:,1]>=0)&(pixel[:,1]<360)
    foreground_safe=np.zeros(len(pixel),bool);foreground_safe[inside]=masks[1][pixel[inside,1],pixel[inside,0]]>0
    keep=(ok.ravel()>0)&(reverse.ravel()>0)&(np.linalg.norm(back[:,0]-p[:,0],axis=1)<1)&foreground_safe
    p=p[keep,0];q=q[keep,0]
    if len(p)<16:return None,dict(status='unsupported',reason='few consistent unoccluded pitch features')
    cv2.setRNGSeed(11026);h,inliers=cv2.findHomography(q,p,cv2.RANSAC,2)
    if h is None or inliers is None:return None,dict(status='unsupported',reason='no ground homography')
    mask=inliers.ravel().astype(bool);count=int(mask.sum());fraction=float(mask.mean())
    extent=cv2.contourArea(cv2.convexHull(q[mask].astype(np.float32)))/(640*360) if count>=3 else 0
    diagnostic=dict(inliers=count,inlier_fraction=fraction,extent=float(extent),features=len(p))
    if count<16 or fraction<.6 or extent<.015:
        checks=[]
        if count<16:checks.append('too few inliers')
        if fraction<.6:checks.append('low inlier consistency')
        if extent<.015:checks.append('narrow spatial extent')
        return None,dict(status='unsupported',reason=', '.join(checks),**diagnostic)
    transform=np.diag([640.,360.,1.]);normal=np.linalg.inv(transform)@h@transform
    corners=np.array([[.05,.05,1],[.95,.05,1],[.95,.95,1],[.05,.95,1]])@normal.T
    denominator=corners[:,2]
    if not np.isfinite(normal).all() or (denominator>0).any() and (denominator<0).any():return None,dict(status='unsupported',reason='projective pole',**diagnostic)
    return normal,dict(status='measured',basis='Current/past white ground features outside actual detection boxes; no person labels or future observations',**diagnostic)
