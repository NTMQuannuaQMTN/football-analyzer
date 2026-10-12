import numpy as np

from cv.refine_calibration import refine


def test_refiner_keeps_original_when_no_pitch_evidence_exists():
    h=np.array([[.1,0,0],[0,.1,0],[0,0,1.]])
    result,diagnostic=refine(np.zeros((720,1280,3),np.uint8),h)
    assert np.array_equal(result,h) and not diagnostic['accepted']
