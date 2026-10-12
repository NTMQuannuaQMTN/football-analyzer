"""Explicit observation timestamps when reducing detections to 1 Hz."""
from collections import defaultdict

import numpy as np


def sample_seconds(rows,duration,policy='legacy_mean',tolerance=.25,causal=False):
    """Rows (t,x,y) -> positions and actual source time at each integer second.

    Nearest sampling preserves a measured location; its source time may differ
    by up to tolerance. Causal mode never uses a future sample. Legacy mean is
    retrospective bucket averaging and labels its mean source timestamp.
    """
    if policy not in ('legacy_mean','nearest'):raise ValueError('Unknown sampling policy')
    if policy=='legacy_mean' and causal:raise ValueError('Bucket mean uses future samples; not a live policy')
    groups=defaultdict(list)
    for t,x,y in rows:
        if not np.isfinite([t,x,y]).all():continue
        sec=int(t) if policy=='legacy_mean' else int(np.floor(t+.5))
        if 0<=sec<=duration:
            if policy=='nearest' and (abs(t-sec)>tolerance+1e-9 or (causal and t>sec+1e-9)):continue
            groups[sec].append((float(t),float(x),float(y)))
    positions={};timestamps={}
    for sec,values in groups.items():
        if policy=='legacy_mean':
            mean=np.mean(values,axis=0);timestamps[sec]=float(mean[0]);positions[sec]=mean[1:]
        else:
            chosen=min(values,key=lambda v:(abs(v[0]-sec),v[0]))
            timestamps[sec]=chosen[0];positions[sec]=np.asarray(chosen[1:])
    return positions,timestamps
