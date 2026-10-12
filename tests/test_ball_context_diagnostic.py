from cv.audit_ball_context import context_sources, longest_missing_run


def test_ball_gap_labels_do_not_call_interpolated_or_held_positions_observed():
    accepted = [(1.8, 5, 6, .8), (4.2, 8, 9, .8)]
    assert context_sources(accepted, 6) == [
        'initial_midfield_fallback', 'accepted_detection_bucket',
        'future_interpolation', 'future_interpolation',
        'accepted_detection_bucket', 'held_previous_coordinate',
        'held_previous_coordinate']


def test_main_camera_missing_run_breaks_at_camera_cut_or_actual_acceptance():
    assert longest_missing_run([(True, False), (True, False), (False, False),
                                (True, False), (True, True), (True, False)]) == 2
