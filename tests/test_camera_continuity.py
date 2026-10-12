import numpy as np
import pytest

from cv.camera_continuity import CameraContinuity


def test_camera_jump_is_rejected_under_past_current_consistent_ground_motion():
    camera=CameraContinuity();base=np.array([[105.,0,0],[0,68.,0],[0,0,1]])
    camera.step(0.,base,None)
    wrong=base.copy();wrong[0,2]=80.
    corrected,d=camera.step(.2,wrong,np.eye(3))
    np.testing.assert_allclose(corrected/corrected[2,2],base)
    assert d['anchor_rejected'] and d['anchor_age_s']==.2
    assert d['anchor_disagreement_m']==80.


def test_camera_history_expires_and_resets_at_unsupported_views():
    camera=CameraContinuity();camera.step(0,np.eye(3),None)
    assert camera.step(3.1,None,np.eye(3))[0] is None
    camera.step(4.,np.eye(3),None)
    assert camera.step(4.2,None,None,supported_view=False)[0] is None
    assert camera.step(4.4,None,np.eye(3))[0] is None
    with pytest.raises(ValueError,match='increasing'):camera.step(4.3,np.eye(3),None)
