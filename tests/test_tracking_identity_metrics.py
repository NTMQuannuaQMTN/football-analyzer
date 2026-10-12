from cv.audit_tracking import identity_counts


def test_global_identity_assignment_penalises_fragmentation():
    result=identity_counts({'p':4},{10:2,20:2},{('p',10):2,('p',20):2})
    assert result=={'idtp':2,'idfn':2,'idfp':2}


def test_identity_assignment_uses_all_valid_edges_not_frame_matching():
    result=identity_counts({'a':3,'b':3},{1:3,2:3},{('a',1):3,('a',2):1,('b',1):1,('b',2):3})
    assert result=={'idtp':6,'idfn':0,'idfp':0}
