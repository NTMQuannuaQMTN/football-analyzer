import numpy as np

from cv.audit_hidden_visibility import footprint_membership


def test_missing_camera_is_distinct_from_outside_and_nonfinite_positions():
    positions = np.array([[[5., 5.], [15., 5.], [np.nan, np.nan]],
                          [[5., 5.], [15., 5.], [np.nan, np.nan]]])
    quads = np.array([[[0., 0.], [10., 0.], [10., 10.], [0., 10.]],
                      [[np.nan, np.nan]] * 4])
    inside, valid = footprint_membership(positions, quads)
    np.testing.assert_equal(valid, [True, False])
    np.testing.assert_equal(inside, [[True, False, False], [False, False, False]])
