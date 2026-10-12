"""Pitch-focused camera motion for BoT-SORT; no player positions or future frames."""
import cv2
import numpy as np


class BroadcastMotion:
    def __init__(self):
        self.method = 'guarded_pitch_flow'
        self.reset_params()

    def reset_params(self):
        self.previous = None
        self.points = None
        self.last_diagnostic = {'accepted': False, 'reason': 'first frame'}

    def apply(self, frame, detections=None):
        small = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        mask = ((hsv[...,0]>30)&(hsv[...,0]<90)&(hsv[...,1]>45)&(hsv[...,2]>35)).astype(np.uint8)*255
        mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((15,15),np.uint8))
        if detections is not None:
            for box in np.asarray(detections).reshape(-1,4):
                x1,y1,x2,y2=(box/2).astype(int)
                cv2.rectangle(mask,(x1-3,y1-3),(x2+3,y2+3),0,-1)
        points = cv2.goodFeaturesToTrack(gray,400,.01,10,mask=mask)
        warp = np.eye(2,3,dtype=float)
        self.last_diagnostic = {'accepted':False,'reason':'insufficient correspondences'}
        if self.previous is not None and self.points is not None and len(self.points)>=12:
            next_points,status,_ = cv2.calcOpticalFlowPyrLK(self.previous,gray,self.points,None)
            if next_points is not None:
                back,back_status,_ = cv2.calcOpticalFlowPyrLK(gray,self.previous,next_points,None)
                if back is not None:
                    valid=status.ravel().astype(bool)&back_status.ravel().astype(bool)&(np.linalg.norm(back-self.points,axis=2).ravel()<1.5)
                    a=self.points[valid].reshape(-1,2); b=next_points[valid].reshape(-1,2)
                    spread=np.ptp(a,axis=0) if len(a) else [0,0]
                    if len(a)>=12 and spread[0]>gray.shape[1]*.2 and spread[1]>gray.shape[0]*.12:
                        candidate,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=2)
                        if candidate is not None:
                            scale=np.linalg.norm(candidate[:,0]); angle=abs(np.arctan2(candidate[1,0],candidate[0,0]))
                            fraction=float(inliers.mean())
                            if .9<scale<1.1 and angle<.12 and fraction>=.5 and np.linalg.norm(candidate[:,2])<gray.shape[1]*.15:
                                warp=candidate;warp[:,2]*=2
                                self.last_diagnostic={'accepted':True,'inliers':int(inliers.sum()),'fraction':fraction,'warp':warp.tolist()}
                            else:
                                self.last_diagnostic={'accepted':False,'reason':'implausible or weak warp','fraction':fraction}
        self.previous=gray;self.points=points
        return warp


def association_cost(previous, current, warp):
    """Robust movement cost on detected feet; no named identities or reference labels."""
    from scipy.optimize import linear_sum_assignment
    previous=np.asarray(previous,float).reshape(-1,4);current=np.asarray(current,float).reshape(-1,4)
    if min(len(previous),len(current))<5:
        return np.inf
    feet=lambda b:np.column_stack([(b[:,0]+b[:,2])/2,b[:,3]])
    a=feet(previous)@warp[:,:2].T+warp[:,2];b=feet(current)
    distance=np.linalg.norm(a[:,None]-b[None,:],axis=2)
    # Scale by observed body height to compare different broadcast zooms.
    scale=np.maximum(10,(previous[:,3]-previous[:,1])[:,None]+(current[:,3]-current[:,1])[None,:])/2
    cost=distance/scale
    rows,cols=linear_sum_assignment(cost)
    return float(np.median(cost[rows,cols]))


class AssociationGuardedMotion:
    """Reject camera warps that worsen same-frame detection association.

    Independent image motion supplies the proposal. Feet supply a consistency
    guard; they never become a calibration target. This remains an experiment
    until held-out tracking tests demonstrate useful behaviour.
    """
    def __init__(self):
        from ultralytics.trackers.utils.gmc import GMC
        self.legacy=GMC(method='sparseOptFlow')
        self.method='association_guarded_flow'
        self.reset_params()

    def reset_params(self):
        self.previous_boxes=None
        self.legacy.reset_params()
        self.last_diagnostic={}

    def apply(self,frame,detections=None):
        proposal=self.legacy.apply(frame,detections)
        identity=np.eye(2,3)
        current=np.asarray(detections).reshape(-1,4) if detections is not None else np.empty((0,4))
        chosen=proposal
        if self.previous_boxes is not None:
            proposed_cost=association_cost(self.previous_boxes,current,proposal)
            identity_cost=association_cost(self.previous_boxes,current,identity)
            # Require a material reduction before overriding the image estimate.
            if np.isfinite(identity_cost) and identity_cost<proposed_cost*.8:
                chosen=identity
            self.last_diagnostic={'proposal_cost':proposed_cost,'identity_cost':identity_cost,'rejected':chosen is identity}
        self.previous_boxes=current.copy()
        return chosen
