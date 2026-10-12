"""Current-image geometric camera fitting; labels/feet are not inference inputs."""
import numpy as np
from itertools import combinations
from scipy.optimize import least_squares

from cv.pitch_lines import LINES, homography_from_lines


def segment_bounds(name):
    """Finite field-marking segment; infinite incidence alone is ambiguous."""
    a,b,c=LINES[name]
    if a:
        low,high=(13.84,54.16) if name.startswith('Big rect.') else (24.84,43.16) if name.startswith('Small rect.') else (0.,68.)
        return np.array([[-c/a,low],[-c/a,high]],float)
    low,high=0.,105.
    if name.startswith('Big rect.'):
        low,high=(0.,16.5) if '. left ' in name else (88.5,105.)
    elif name.startswith('Small rect.'):
        low,high=(0.,5.5) if '. left ' in name else (99.5,105.)
    return np.array([[low,-c/b],[high,-c/b]],float)


def segment_distances(world,name):
    a,b=segment_bounds(name);direction=b-a
    fraction=np.clip((world-a)@direction/(direction@direction),0,1)
    nearest=a+fraction[:,None]*direction
    return np.linalg.norm(world-nearest,axis=1)


def image_line_residuals(image_to_pitch, points):
    """Signed distance in normalised image units, equal total weight per line."""
    residual=[]
    for name,xy in points.items():
        if name not in LINES or len(xy)<2:
            continue
        xy=np.asarray(xy,float)
        if not np.isfinite(xy).all():continue
        line=np.asarray(LINES[name],float)@image_to_pitch
        norm=np.linalg.norm(line[:2])
        if norm<1e-12:continue
        d=(np.column_stack([xy,np.ones(len(xy))])@line)/norm
        residual.extend(d/np.sqrt(len(xy)))
    return np.asarray(residual)


def fit_camera(points, robust=True):
    """Fit line incidence in image space after the algebraic initializer.

    Each named line contributes equally; robust loss limits point annotation
    noise. An image fit residual is a diagnostic, NOT metre-error confidence.
    """
    initial=homography_from_lines(points)
    if initial is None:return None,dict(supported=False,reason='insufficient independent markings')
    diagnostic=dict(supported=True,refined=False,lines=len([n for n,p in points.items() if n in LINES and len(p)>=2]))
    before=image_line_residuals(initial,points)
    diagnostic['initial_rms_image_units']=float(np.sqrt(np.mean(before**2)))
    if not robust:return initial,diagnostic
    normalise=np.diag([1/105,1/68,1])
    matrix=normalise@initial
    if abs(matrix[2,2])<1e-10:return initial,diagnostic
    matrix=matrix/matrix[2,2]
    def unpack(p):return np.diag([105,68,1])@np.r_[p,1].reshape(3,3)
    p0=matrix.ravel()[:8]
    try:
        result=least_squares(lambda p:image_line_residuals(unpack(p),points),p0,
                             loss='soft_l1',f_scale=.002,max_nfev=100)
        h=unpack(result.x)
        after=image_line_residuals(h,points)
        if not np.isfinite(h).all() or abs(np.linalg.det(h))<1e-12:
            return initial,diagnostic
        # Safety: no image-alignment regression. Metre errors remain independent.
        diagnostic['final_rms_image_units']=float(np.sqrt(np.mean(after**2)))
        if diagnostic['final_rms_image_units']<=diagnostic['initial_rms_image_units']*1.01:
            diagnostic['refined']=True
            return h,diagnostic
    except (ValueError,np.linalg.LinAlgError):pass
    return initial,diagnostic


def consensus_camera(points, threshold=.004):
    """Reject inconsistent named image lines with deterministic minimal fits.

    All proposals and scores use current image markings only. Wrong names and
    underconstrained views may remain ambiguous; consensus is not confidence.
    """
    clean={n:np.asarray(p,float) for n,p in points.items() if n in LINES and len(p)>=2 and np.isfinite(p).all()}
    xs=[n for n in clean if LINES[n][0]];ys=[n for n in clean if LINES[n][1]]
    best=None;best_key=None
    for xp in combinations(xs,2):
        if LINES[xp[0]]==LINES[xp[1]]:continue
        for yp in combinations(ys,2):
            if LINES[yp[0]]==LINES[yp[1]]:continue
            proposal=homography_from_lines({n:clean[n] for n in xp+yp})
            if proposal is None:continue
            distances={}
            for name,xy in clean.items():
                line=np.asarray(LINES[name])@proposal;norm=np.linalg.norm(line[:2])
                distances[name]=float(np.median(np.abs(np.column_stack([xy,np.ones(len(xy))])@line))/max(norm,1e-12))
            names=[]
            for n,d in distances.items():
                if d>threshold:continue
                xy=clean[n];hp=np.column_stack([xy,np.ones(len(xy))])@proposal.T
                world=hp[:,:2]/hp[:,2:]
                if np.isfinite(world).all() and np.median(segment_distances(world,n))<=3:
                    names.append(n)
            unique={LINES[n] for n in names}
            if len({c/a for a,b,c in unique if a})<2 or len({c/b for a,b,c in unique if b})<2:
                continue
            key=(len(unique),len(names),-sum(min(d,threshold) for d in distances.values()))
            if best_key is None or key>best_key:
                best_key=key;best=(proposal,names,distances)
    if best is None:return None,dict(supported=False,reason='no independent rectangle')
    proposal,names,distances=best
    # The selected subset is fit algebraically to avoid the catastrophic tails
    # observed in the unconstrained nonlinear image-fit experiment.
    refit=homography_from_lines({n:clean[n] for n in names})
    if refit is None:refit=proposal
    return refit,dict(supported=True,consensus_lines=names,rejected_lines=[n for n in clean if n not in names],
        unique_consensus_lines=best_key[0],threshold_image_units=threshold,
        image_line_distances=distances,confidence='not calibrated; marking agreement only')
