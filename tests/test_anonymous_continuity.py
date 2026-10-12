from cv.audit_anonymous_continuity import measure


def row(t,identifier,shot,x,live=True):
    return dict(t=t,live=live,dets=[dict(id=identifier,shot_id=shot,cls='person',px=x,py=20.)])


def test_anonymous_movement_never_joins_resets_misses_or_other_people():
    rows=[row(0,1,0,10),row(.2,1,0,11),row(.4,2,0,50),
          row(.6,2,1,80),row(.8,2,1,81,live=False),row(1.,2,1,82)]
    result=measure(rows,.2)
    assert result['all']['n']==1
    assert result['all']['median_m_s']==5
    assert result['all']['steps_over_10m_s']==0
