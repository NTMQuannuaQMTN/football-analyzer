import numpy as np
from backend.reconstruction.replay_bridge import bridge

def test_model_corrections_cannot_break_a_feasible_next_sighting_connection():
    p=np.array([[[0.,0]],[[90,60]],[[80,60]],[[70,60]],[[20,0]]]);v=np.array([[True],[False],[False],[False],[True]])
    out,r=bridge(p,v,vmax=8)
    assert np.array_equal(out[v],p[v]) and np.array_equal(p[1],[[90,60]])
    assert (np.linalg.norm(np.diff(out[:,0],axis=0),axis=1)<=8+1e-8).all()
    assert r['feasible_gaps']==1 and not r['conflicts']

def test_future_sighting_changes_replay_and_impossible_identity_is_not_repaired():
    p=np.array([[[0.,0]],[[2,4]],[[4,0]]]);v=np.array([[True],[False],[True]])
    a,_=bridge(p,v,vmax=2);p[-1]=[[0,0]];b,_=bridge(p,v,vmax=2)
    assert not np.allclose(a[1],b[1])
    p[-1]=[[100,0]];c,r=bridge(p,v,vmax=2)
    assert np.array_equal(c,p) and len(r['conflicts'])==1

def test_off_pitch_gap_is_not_connected_and_leading_gap_uses_future():
    p=np.array([[[50.,0]],[[50.,0]],[[0,0]],[[100,0]]]);v=np.array([[False],[False],[True],[True]])
    o=np.array([[True],[True],[True],[False]])
    out,r=bridge(p,v,o,vmax=5)
    assert out[1,0,0]==5 and out[0,0,0]==10
    assert out[-1,0,0]==100 and not r['conflicts']
