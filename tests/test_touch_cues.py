import numpy as np

from cv.touch_cues import apply_touch_cues


def test_touch_cues_do_not_invent_observations_or_move_measured_feet():
    positions=np.array([[[10.,10.]],[[20.,20.]]]); visible=np.array([[True],[False]])
    result=apply_touch_cues(positions,visible,{(0,0):(30,30),(1,0):(24,24)})
    assert np.array_equal(result[0],positions[0])
    assert np.allclose(result[1],[[21,21]])
    assert np.array_equal(visible,[[True],[False]]) and np.array_equal(positions[1],[[20,20]])
