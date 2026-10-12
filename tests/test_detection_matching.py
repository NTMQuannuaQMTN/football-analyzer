import numpy as np

from cv.evaluate_session import iou_matrix, matched_pairs


def test_box_matching_does_not_match_nearby_nonoverlapping_players():
    values=iou_matrix([[0,0,10,20],[20,0,30,20]],[[20,0,30,20],[0,0,10,20]])
    assert np.array_equal(values,[[0,1],[1,0]])
    assert iou_matrix([],[[0,0,10,20]]).shape==(0,1)


def test_threshold_matching_preserves_two_valid_crowded_pairs():
    # Maximum summed IoU chooses .99+.49 and loses a valid pair after cutoff.
    matrix=np.array([[.99,.70],[.60,.49]])
    assert set(matched_pairs(matrix))=={(0,1),(1,0)}
    assert matched_pairs(np.empty((0,2)))==[]
