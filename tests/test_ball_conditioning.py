import numpy as np

from cv.audit_ball_conditioning import context


def test_context_missingness_stresses_are_causal_and_do_not_mutate_reference():
    ball=np.arange(120,dtype=float).reshape(60,2);possession=np.arange(60)%2
    before=ball.copy();before_possession=possession.copy()
    stale,_=context(ball,possession,'ball_observed_every_30s')
    changed=ball.copy();changed[21:]=1000
    other,_=context(changed,possession,'ball_observed_every_30s')
    np.testing.assert_array_equal(stale[:21],other[:21])
    np.testing.assert_array_equal(stale[29],ball[0])
    np.testing.assert_array_equal(stale[30],ball[30])
    fallback,unknown=context(ball,possession,'midfield_ball_and_possession_unknown')
    np.testing.assert_array_equal(fallback,np.tile([52.5,34.],(60,1)))
    assert (unknown==-1).all()
    np.testing.assert_array_equal(ball,before);np.testing.assert_array_equal(possession,before_possession)
