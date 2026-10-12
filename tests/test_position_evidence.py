from types import SimpleNamespace

import numpy as np

from backend.tactics.position_evidence import geometry,position_evidence,reason_from_positions


def test_geometry_and_radius_envelope_contain_perturbed_shapes():
    points=np.column_stack([np.arange(10)*5+20,np.arange(10)*4+10])
    m=SimpleNamespace(T=1,order=list(range(10)),frames=[dict(t=0,obs=[False]*10,rad=[3]*10)],
                      team_mask=lambda team:np.ones(10,bool),own_xy=lambda team:points[None])
    ev=position_evidence(m,"A",0)
    rng=np.random.default_rng(0)
    for _ in range(30):
        directions=rng.normal(size=(10,2)); directions/=np.linalg.norm(directions,axis=1)[:,None]
        shifted=points+directions*rng.uniform(0,3,(10,1))
        for key,value in geometry(shifted).items():
            low,high=ev['sensitivity_bounds_m'][key]
            assert low-.1<=value<=high+.1


def test_unknown_radius_is_not_replaced_with_confidence():
    p=np.column_stack([np.arange(10)*5+20,np.arange(10)*4+10])
    m=SimpleNamespace(T=1,order=list(range(10)),frames=[dict(t=0,obs=[False]*10)],
                      team_mask=lambda team:np.ones(10,bool),own_xy=lambda team:p[None])
    ev=position_evidence(m,"A",0)
    assert ev['sensitivity_bounds_m'] is None and ev['uncertainty']=='uncalibrated'
    reason=reason_from_positions(dict(deltas={'def_line_oop_m':5},outcome={'type':'GOAL'}),ev)
    assert 'Tentative' in reason['interpretation'] and 'may' in reason['mechanism']
    assert reason['observed_outcome']=='GOAL' and reason['retrospective'] is True
