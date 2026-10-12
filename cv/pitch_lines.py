"""Learn named pitch markings from local GSR labels, independently of player identities.

The line-to-pitch solver abstains unless at least four independent ground
lines constrain the homography. Player coordinates are evaluation-only.
"""
import cv2
import numpy as np
import torch
from torch import nn


# World coordinates in metres. Goal frames are above the pitch and excluded.
LINES = {
    "Side line top": (0, 1, 0), "Side line bottom": (0, 1, -68),
    "Side line left": (1, 0, 0), "Side line right": (1, 0, -105),
    "Middle line": (1, 0, -52.5),
    "Big rect. left main": (1, 0, -16.5),
    "Big rect. right main": (1, 0, -88.5),
    "Big rect. left top": (0, 1, -13.84),
    "Big rect. right top": (0, 1, -13.84),
    "Big rect. left bottom": (0, 1, -54.16),
    "Big rect. right bottom": (0, 1, -54.16),
    "Small rect. left main": (1, 0, -5.5),
    "Small rect. right main": (1, 0, -99.5),
    "Small rect. left top": (0, 1, -24.84),
    "Small rect. right top": (0, 1, -24.84),
    "Small rect. left bottom": (0, 1, -43.16),
    "Small rect. right bottom": (0, 1, -43.16),
}
NAMES = tuple(LINES)


def homography_from_lines(points):
    """Normalized image points (0..1) -> pitch metres, or None.

    Each point must lie on its named line. SVD solves l^T H p = 0.
    Rank checks prevent arbitrary extrapolation from a single touchline.
    """
    rows = []
    used = set()
    for name, xy in points.items():
        if name not in LINES or len(xy) < 2:
            continue
        xy = np.asarray(xy)
        if not np.isfinite(xy).all():
            continue
        a,b,c = LINES[name]
        line = np.array([a*105,b*68,c],float)
        line /= np.linalg.norm(line[:2])
        used.add(tuple(LINES[name]))
        for x,y in xy:
            rows.append(np.outer(line,[x,y,1]).ravel())
    # Four labels are not sufficient when three world lines are concurrent
    # (e.g. three parallel goal/box fronts and just one touchline). Annotation
    # noise can make numerical SVD rank appear full despite this structural
    # ambiguity and produce a horizon through the pitch. Require a rectangle
    # from two distinct members of each pitch-axis line family.
    vertical = {c / a for a,b,c in used if a and not b}
    horizontal = {c / b for a,b,c in used if b and not a}
    if len(vertical)<2 or len(horizontal)<2 or len(rows) < 8:
        return None
    _,s,v = np.linalg.svd(np.asarray(rows),full_matrices=len(rows)<9)
    if len(s)<8 or s[7] < s[0]*1e-5:
        return None
    h = v[-1].reshape(3,3)
    if abs(np.linalg.det(h))<1e-8:
        return None
    return np.diag([105,68,1]) @ h


def heatmap_points(prob, threshold=.45):
    """Robust image-line fit; isolated speckles do not define a camera."""
    result = {}
    for name,p in zip(NAMES,prob):
        y,x = np.where(p>threshold)
        if len(x)<15:
            continue
        xy = np.column_stack([x/(p.shape[1]-1), y/(p.shape[0]-1)]).astype(np.float32)
        vx,vy,cx,cy = cv2.fitLine(xy,cv2.DIST_HUBER,0,.01,.01).ravel()
        t = (xy-[cx,cy]) @ [vx,vy]
        lo,hi = np.percentile(t,[5,95])
        if hi-lo<.06:
            continue
        result[name] = np.asarray([[cx+lo*vx,cy+lo*vy],[cx+hi*vx,cy+hi*vy]])
    return result


class PitchLineNet(nn.Module):
    """Small U-net; keeps geometry at quarter input resolution."""
    def __init__(self):
        super().__init__()
        def block(a,b,stride=1):
            return nn.Sequential(nn.Conv2d(a,b,3,stride,1),nn.GroupNorm(4,b),nn.SiLU(),
                                 nn.Conv2d(b,b,3,1,1),nn.GroupNorm(4,b),nn.SiLU())
        self.a = block(3,16,2)
        self.b = block(16,32,2)
        self.c = block(32,64,2)
        self.d = block(64,96,2)
        self.up1 = block(160,64)
        self.up2 = block(96,32)
        self.head = nn.Conv2d(32,len(NAMES),1)

    def forward(self,x):
        a=self.a(x); b=self.b(a); c=self.c(b); d=self.d(c)
        z=self.up1(torch.cat([torch.nn.functional.interpolate(d,size=c.shape[-2:],mode="bilinear",align_corners=False),c],1))
        z=self.up2(torch.cat([torch.nn.functional.interpolate(z,size=b.shape[-2:],mode="bilinear",align_corners=False),b],1))
        return self.head(z)
