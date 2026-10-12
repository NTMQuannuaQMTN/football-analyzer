"""Conservative anonymous appearance association with spatial/ambiguity gates."""
import numpy as np
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.utils import matching


def appearance_costs(base,similarity,proximity,minimum=.9,margin=.05):
    """Only mutually unique, sufficiently distinct nearby matches lower cost."""
    if not base.size:return base.copy()
    eligible=proximity&np.isfinite(similarity)&(similarity>=minimum)
    score=np.where(eligible,similarity,-2.)
    result=base.copy()
    for r in range(len(base)):
        c=int(np.argmax(score[r]))
        if not eligible[r,c] or int(np.argmax(score[:,c]))!=r:continue
        row_other=np.delete(score[r],c);col_other=np.delete(score[:,c],r)
        if len(row_other) and score[r,c]-row_other.max()<margin:continue
        if len(col_other) and score[r,c]-col_other.max()<margin:continue
        result[r,c]=min(result[r,c],(1-similarity[r,c])/2)
    return result


class AppearanceBOTSORT(BOTSORT):
    def __init__(self,args):
        super().__init__(args)
        # Missing crops retain no appearance evidence rather than a shared
        # fabricated feature vector. Native update slices features by score.
        self.encoder=lambda feats,_:[f if np.linalg.norm(f)>1e-8 else None for f in feats]

    def get_dists(self,tracks,detections):
        base=matching.iou_distance(tracks,detections)
        proximity=base<=1-self.args.proximity_thresh
        if self.args.fuse_score:base=matching.fuse_score(base,detections)
        if not base.size:return base
        similarity=np.full(base.shape,np.nan)
        for i,t in enumerate(tracks):
            if t.smooth_feat is None:continue
            for j,d in enumerate(detections):
                if d.curr_feat is not None:similarity[i,j]=np.clip(t.smooth_feat@d.curr_feat,-1,1)
        return appearance_costs(base,similarity,proximity,self.args.minimum_cosine,self.args.appearance_margin)
