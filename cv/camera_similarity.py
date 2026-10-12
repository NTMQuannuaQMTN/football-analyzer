"""Experimental constrained ground-flow fallback for narrow painted regions.

Fewer degrees of freedom than a homography, but perspective changes can make
this approximation wrong. Validation must precede any production use.
"""
import cv2
import numpy as np

from cv.camera_motion import ground_mask, ground_motion


def similarity_fallback(previous,current):
    h,diagnostic=ground_motion(previous,current)
    if h is not None:return h,diagnostic
    # A weaker spatial span is admissible only for a bounded four-parameter
    # similarity, never for an unconstrained perspective transform.
    size=(640,360);a=cv2.resize(previous,size);b=cv2.resize(current,size)
    ga=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY);gb=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)
    p=cv2.goodFeaturesToTrack(ga,400,.01,8,mask=ground_mask(a))
    if p is None or len(p)<16:return None,diagnostic
    q,ok,_=cv2.calcOpticalFlowPyrLK(ga,gb,p,None,winSize=(25,25),maxLevel=3)
    if q is None:return None,diagnostic
    back,reverse,_=cv2.calcOpticalFlowPyrLK(gb,ga,q,None,winSize=(25,25),maxLevel=3)
    if back is None:return None,diagnostic
    keep=(ok.ravel()>0)&(reverse.ravel()>0)&(np.linalg.norm(back[:,0]-p[:,0],axis=1)<1)
    p=p[keep,0];q=q[keep,0]
    if len(p)<16:return None,diagnostic
    cv2.setRNGSeed(11026)
    affine,inliers=cv2.estimateAffinePartial2D(q,p,method=cv2.RANSAC,ransacReprojThreshold=2)
    if affine is None or inliers is None:return None,diagnostic
    inside=inliers.ravel().astype(bool);count=int(inside.sum());fraction=float(inside.mean())
    extent=cv2.contourArea(cv2.convexHull(q[inside].astype(np.float32)))/(640*360) if count>=3 else 0
    scale=float(np.linalg.norm(affine[:,0]));angle=abs(float(np.arctan2(affine[1,0],affine[0,0])))
    shift=float(np.linalg.norm(affine[:,2]))
    if count<16 or fraction<.6 or extent<.003 or not .9<scale<1.1 or angle>=.12 or shift>=640*.15:
        return None,dict(status='unsupported',reason='homography and bounded similarity unsupported',homography=diagnostic,
            similarity_inliers=count,similarity_fraction=fraction,similarity_extent=float(extent))
    matrix=np.vstack([affine,[0,0,1.]])
    transform=np.diag([640.,360.,1.]);normal=np.linalg.inv(transform)@matrix@transform
    return normal,dict(status='measured_similarity',inliers=count,inlier_fraction=fraction,extent=float(extent),scale=scale,rotation_rad=angle,
        basis='Current/past white ground points; bounded similarity approximation after homography abstention; not positional confidence')
