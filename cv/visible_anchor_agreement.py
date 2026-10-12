"""Camera agreement at actual visible feet; never reads reference pitch points."""
import numpy as np

from cv.gsr import project


def visible_foot_disagreement(propagated,anchor,detections,image_size):
    detections=np.asarray(detections,float)
    if detections.size==0:return None
    if detections.ndim!=2 or detections.shape[1]<5:raise ValueError('Detections require box coordinates and confidence')
    width,height=image_size
    if width<=0 or height<=0:raise ValueError('Image size must be positive')
    selected=detections[np.isfinite(detections[:,:5]).all(1)&(detections[:,4]>=.3)]
    feet=np.column_stack([(selected[:,0]+selected[:,2])/2/width,selected[:,3]/height])
    feet=feet[(feet>=0).all(1)&(feet<=1).all(1)]
    if len(feet)<4:return None
    displacement=np.linalg.norm(project(propagated,feet)-project(anchor,feet),axis=1)
    if not np.isfinite(displacement).all():return float('inf')
    return float(np.percentile(displacement,90))
