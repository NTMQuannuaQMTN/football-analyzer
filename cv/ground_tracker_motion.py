"""Conservative ground-paint motion for an optional anonymous tracker audit."""
import cv2
import numpy as np

from cv.camera_motion import ground_motion,ground_mask


def affine_from_ground(previous,current):
    h,diagnostic=ground_motion(previous,current)
    if h is None:return np.eye(2,3),diagnostic
    small=cv2.resize(current,(640,360));y,x=np.nonzero(ground_mask(small))
    if len(x)<30:return np.eye(2,3),dict(status='unsupported',reason='few current paint pixels')
    indices=np.linspace(0,len(x)-1,min(400,len(x)),dtype=int)
    q=np.column_stack([x[indices]/640,y[indices]/360])
    hp=np.column_stack([q,np.ones(len(q))])@h.T
    with np.errstate(divide='ignore',invalid='ignore'):p=hp[:,:2]/hp[:,2:]
    valid=np.isfinite(p).all(1)&(p>0).all(1)&(p<1).all(1)
    if valid.sum()<20:return np.eye(2,3),dict(status='unsupported',reason='ground overlap too small')
    # Native tracker GMC expects previous pixels -> current pixels.
    scale=np.array([current.shape[1],current.shape[0]])
    p=p[valid]*scale;q=q[valid]*scale
    matrix,inliers=cv2.estimateAffinePartial2D(p,q,method=cv2.RANSAC,ransacReprojThreshold=3*scale[0]/640,maxIters=1000)
    if matrix is None or inliers.mean()<.6:return np.eye(2,3),dict(status='unsupported',reason='ground affine fit weak')
    singular=np.linalg.svd(matrix[:,:2],compute_uv=False)
    residual=np.linalg.norm(np.column_stack([p,np.ones(len(p))])@matrix.T-q,axis=1)*640/scale[0]
    if singular.min()<.5 or singular.max()>2 or np.median(residual)>2:
        return np.eye(2,3),dict(status='unsupported',reason='ground affine approximation implausible')
    return matrix,dict(status='accepted',paint_points=len(p),inlier_fraction=float(inliers.mean()),median_residual_px_640=float(np.median(residual)))
