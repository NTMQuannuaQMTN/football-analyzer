"""Ball-touch estimates are soft cues, not new player observations."""
import numpy as np


def apply_touch_cues(positions,visible,pins,weight=.25):
    result=np.array(positions,copy=True)
    for (t,j),point in pins.items():
        if visible[t,j] or not np.isfinite(result[t,j]).all():
            continue
        result[t,j]=(1-weight)*result[t,j]+weight*np.asarray(point)
    return result
