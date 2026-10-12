"""Retrospective feasible paths between observations; never repairs measured anchors."""
import numpy as np


def _fraction(base,target,center,radius):
    d=target-base;q=base-center;a=float(d@d)
    if a<1e-12:return 1.
    b=2*float(q@d);c=float(q@q)-radius*radius
    return float(np.clip((-b+np.sqrt(max(0.,b*b-4*a*c)))/(2*a),0,1))


def bridge(positions,visible,on=None,vmax=8.5):
    p=np.array(positions,dtype=float,copy=True);visible=np.asarray(visible,bool)
    if on is None:on=np.isfinite(p).all(2)
    on=np.asarray(on,bool)
    if p.shape[:2]!=visible.shape or visible.shape!=on.shape or not np.isfinite(vmax) or vmax<=0:raise ValueError('Invalid bridge shapes or speed')
    conflicts=[];gaps=0
    for j in range(p.shape[1]):
        # Never connect across substitutions/off-pitch intervals or missing coordinates.
        active=on[:,j]&np.isfinite(p[:,j]).all(1)
        edges=np.diff(np.r_[False,active,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
        for start,end in zip(starts,ends):
            seen=np.flatnonzero(visible[start:end,j])+start
            if not len(seen):continue
            for a,b in zip(seen[:-1],seen[1:]):
                speed=float(np.linalg.norm(p[b,j]-p[a,j])/(b-a))
                if speed>vmax+1e-8:
                    conflicts.append(dict(player_index=j,last_t=int(a),next_t=int(b),required_mean_speed_m_s=speed,reason='Incompatible measured anchors; review camera/identity/time. Anchors preserved.'))
                    continue
                if b-a<=1:continue
                gaps+=1
                for k in range(a+1,b):
                    previous=p[k-1,j];target=p[k,j].copy();remaining=b-k
                    base=previous+(p[b,j]-previous)/(remaining+1)
                    f=min(_fraction(base,target,previous,vmax),_fraction(base,target,p[b,j],remaining*vmax))
                    p[k,j]=base+f*(target-base)
            # Leading/trailing estimates connect to the sole available sighting.
            for k in range(seen[0]-1,start-1,-1):
                d=p[k,j]-p[k+1,j];r=np.linalg.norm(d)
                if r>vmax:p[k,j]=p[k+1,j]+d/r*vmax
            for k in range(seen[-1]+1,end):
                d=p[k,j]-p[k-1,j];r=np.linalg.norm(d)
                if r>vmax:p[k,j]=p[k-1,j]+d/r*vmax
    return p,dict(mode='retrospective',uses_future_sightings=True,speed_limit_m_s=vmax,feasible_gaps=gaps,conflicts=conflicts,
        note='Conditional on identity and calibrated measurements. Feasible motion is not validated position accuracy; incompatible sightings are not moved.')
