"""Current/past image camera continuity; external anchors retain their provenance."""
import numpy as np

from cv.camera_motion import propagate
from cv.gsr import project


class CameraContinuity:
    def __init__(self, max_age_s=3., agreement_m=10.):
        self.max_age_s=max_age_s;self.agreement_m=agreement_m
        self.h=None;self.anchor_t=float('-inf');self.last_t=None

    def step(self,t,anchor,current_to_previous,supported_view=True):
        if not np.isfinite(t) or self.last_t is not None and t<=self.last_t:
            raise ValueError('Camera updates need finite increasing timestamps')
        self.last_t=t
        if not supported_view:
            self.h=None;self.anchor_t=float('-inf')
            return None,dict(status='unsupported',reason='not a current main-camera view')
        h=propagate(self.h,current_to_previous)
        if t-self.anchor_t>self.max_age_s:h=None
        accepted=False;delta=None
        if anchor is not None:
            accepted=h is None or self.agreement_m==0
            if not accepted:
                points=[[.2,.4],[.5,.4],[.8,.4],[.2,.8],[.5,.8],[.8,.8]]
                values=np.linalg.norm(project(h,points)-project(anchor,points),axis=1)
                if np.isfinite(values).all():
                    delta=float(np.median(values));accepted=delta<=self.agreement_m
            if accepted:h=np.array(anchor,copy=True);self.anchor_t=t
        self.h=h
        return h,dict(status='anchor' if accepted else 'propagated' if h is not None else 'unsupported',
            anchor_accepted=accepted,anchor_rejected=anchor is not None and not accepted,
            anchor_disagreement_m=delta,anchor_age_s=t-self.anchor_t if np.isfinite(self.anchor_t) else None,
            reason='No reference coordinates or future images enter this state; external anchor may be retrospective')
