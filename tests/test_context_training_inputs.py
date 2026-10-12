import numpy as np

from cv.train_context_robustness import training_context


def test_context_training_uses_only_past_ball_and_explicit_unknown_possession():
    ball=np.arange(240,dtype=float).reshape(120,2);possession=np.arange(120)%2
    stale,unknown=training_context(ball,possession)
    changed=ball.copy();changed[41:]=1000
    other,_=training_context(changed,possession)
    np.testing.assert_array_equal(stale[:41],other[:41])
    np.testing.assert_array_equal(unknown[:60],possession[:60])
    assert (unknown[60:]==-1).all()
    assert (possession>=0).all()
