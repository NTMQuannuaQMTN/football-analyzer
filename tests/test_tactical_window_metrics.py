import numpy as np

from cv.tactical_window_metrics import comparison_windows,summarize_changes


def test_constant_camera_bias_can_cancel_in_window_changes_but_not_position_error():
    reference=np.tile([40.,30.,25.],(1200,1));reference[600:]+=np.array([5,3,4])
    estimate=reference+[0,0,5]
    result=comparison_windows(reference,estimate,np.ones(1200),'A')
    for rows in result.values():
        assert len(rows)==15
        assert summarize_changes(rows)['mean_absolute_delta_error_m']==0
    assert np.abs(estimate[:,2]-reference[:,2]).mean()==5


def test_false_and_opposite_geometric_changes_are_explicitly_counted():
    rows=[dict(reference_delta_m=0,estimated_delta_m=3),dict(reference_delta_m=4,estimated_delta_m=-3),
          dict(reference_delta_m=3,estimated_delta_m=4),dict(reference_delta_m=3,estimated_delta_m=0)]
    result=summarize_changes(rows)
    assert result['unsupported_threshold_changes']==1
    assert result['opposite_direction_changes']==1
    assert result['same_direction_reference_changes']==1
    assert result['missed_reference_changes']==1
