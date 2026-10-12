import numpy as np

from backend.reconstruction.real_benchmark import camera_centre, mask_k


def test_initial_camera_and_visibility_do_not_use_future_ball_positions():
    ball = np.array([[20., 34.], [25., 34.], [40., 34.], [45., 34.], [50., 34.]])
    changed = ball.copy()
    changed[1:, 0] = 100
    np.testing.assert_equal(camera_centre(ball[:, 0])[:1], camera_centre(changed[:, 0])[:1])
    positions = np.broadcast_to(np.array([[22., 34.], [52., 34.], [80., 34.]]), (5, 3, 2)).copy()
    np.testing.assert_equal(mask_k(dict(pos=positions, ball=ball), 1)[:1],
                            mask_k(dict(pos=positions, ball=changed), 1)[:1])


def test_missing_initial_ball_has_finite_prefix_consistent_camera():
    ball = np.array([np.nan, np.nan, 80., 75.])
    whole = camera_centre(ball)
    for length in range(1, len(ball) + 1):
        np.testing.assert_equal(whole[:length], camera_centre(ball[:length]))
    assert np.isfinite(whole).all()
    assert camera_centre(np.array([])).size == 0
