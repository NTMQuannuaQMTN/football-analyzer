import cv2
import numpy as np

from cv.camera_quality import assess_camera,plausible


def grass():
    return cv2.cvtColor(np.full((144,256,3),(55,170,150),np.uint8),cv2.COLOR_HSV2BGR)


def test_plausibility_is_homogeneous_sign_invariant():
    h=np.diag([105,68,1.])
    a=assess_camera(grass(),h);b=assess_camera(grass(),-h)
    assert plausible(a) and plausible(b)
    assert a['grass_inside_pitch_fraction']==b['grass_inside_pitch_fraction']


def test_horizon_through_grass_is_flagged():
    h=np.array([[105,0,0],[0,68,0],[0,1,-.5]])
    d=assess_camera(grass(),h)
    assert d['horizon_crossing_fraction']>.4 and not plausible(d)


def test_scale_declares_reference_pixel_units_for_resized_images():
    diagnostic=assess_camera(grass(),np.diag([105,68,1.]))
    assert diagnostic['jacobian_reference_image_size']==[1920,1080]
    assert diagnostic['p90_metres_per_reference_pixel']==diagnostic['p90_metres_per_original_pixel']
    assert np.isclose(diagnostic['p90_metres_per_reference_pixel'],max(105/1920,68/1080))
