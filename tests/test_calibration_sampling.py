import numpy as np

from cv.soccernet import Calibration


def test_causal_camera_sampling_never_uses_the_next_estimate():
    camera=Calibration.__new__(Calibration)
    camera.H=[np.eye(3)*i for i in [1,2,3]];camera.conf=[1,1,1];camera.min_conf=.5
    assert np.array_equal(camera.at(.4),camera.H[1])
    assert np.array_equal(camera.at(.4,causal=True),camera.H[0])
    assert np.array_equal(camera.at(.5,causal=True),camera.H[1])
    assert camera.at(-.1,causal=True) is None
    assert camera.at(float('nan')) is None


def test_camera_sample_must_belong_to_the_current_main_shot():
    camera=Calibration.__new__(Calibration)
    camera.H=[np.eye(3),np.eye(3)*2,np.eye(3)*3]
    camera.conf=[1,1,1];camera.min_conf=.5
    camera.main_segments=((0.,.45),(.55,1.1))
    assert camera.at(.4) is None  # nearest .5s is a non-main camera
    np.testing.assert_equal(camera.at(.4,allow_cross_shot=True),camera.H[1])
    np.testing.assert_equal(camera.at(.4,causal=True),camera.H[0])
    assert camera.at(.6,causal=True) is None  # past .5s is from the other view
    assert camera.at(.5) is None  # query itself is unsupported
    np.testing.assert_equal(camera.at(1.),camera.H[2])


def test_neighbouring_main_shots_are_not_blended_or_shared():
    camera=Calibration.__new__(Calibration)
    camera.H=[np.eye(3),np.eye(3)*2];camera.conf=[1,1];camera.min_conf=.5
    camera.main_segments=((0.,.4),(.4,1.))
    assert camera.at(.3) is None  # .5s camera belongs to the next main shot
    np.testing.assert_equal(camera.at(.3,causal=True),camera.H[0])
