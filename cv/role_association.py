"""Role assignment preserves anonymous track continuity when explicitly enabled.

Stable track aliases are not verified shirt identities. Spatial priors choose
initial aliases only; they must not swap an existing track's alias each second.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def assign(cost,slots,previous,gate=60.,stable=False,override=None):
    cost=np.asarray(cost,float)
    if len(set(slots))!=len(slots):raise ValueError('Duplicate anonymous slot in a frame')
    override=override or {};pairs=[];used=set();fixed=set()
    if stable:
        for a,sid in enumerate(slots):
            matching=[i for i,s in enumerate(previous) if s==sid]
            if len(matching)==1:
                i=matching[0]
                if sid in override and override[sid]!=i:continue
                pairs.append((a,i));used.add(i);fixed.add(a)
    rows=[a for a in range(len(slots)) if a not in fixed];cols=[i for i in range(cost.shape[1]) if i not in used]
    if rows and cols:
        r,c=linear_sum_assignment(cost[np.ix_(rows,cols)])
        pairs.extend((rows[a],cols[i]) for a,i in zip(r,c) if cost[rows[a],cols[i]]<=gate)
    return sorted(pairs)
