from backend.reconstruction.measurement_review import projected_motion_review


def test_recent_large_jump_flags_measurements_without_changing_positions_or_using_future():
    frames=[dict(t=0,xy=[[10,20]],obs=[True]),dict(t=2,xy=[[90,20]],obs=[True]),
            dict(t=3,xy=[[10,20]],obs=[True])]
    result=projected_motion_review(frames,0,2)
    assert result['status']=='review_camera_or_identity'
    assert result['apparent_speed_m_s']==40 and result['current_t']==2
    assert frames[1]['xy']==[[90,20]]


def test_inferred_points_and_future_source_measurements_do_not_become_evidence():
    frames=[dict(t=0,xy=[[10,20]],obs=[True]),dict(t=1,xy=[[90,20]],obs=[False]),
            dict(t=2,xy=[[90,20]],obs=[True],observed_t=[2.4])]
    result=projected_motion_review(frames,0,2)
    assert result['status']=='not_enough_recent_observations'
    assert result['apparent_speed_m_s'] is None


def test_source_times_and_stale_gaps_are_explicit_and_small_jump_is_not_accuracy():
    frames=[dict(t=0,xy=[[10,20]],obs=[True],observed_t=[.4]),
            dict(t=1,xy=[[11,20]],obs=[True],observed_t=[1.2])]
    result=projected_motion_review(frames,0,2)
    assert result['apparent_speed_m_s']==1.25 and result['status']=='no_large_recent_jump'
    assert result['timing_basis']==['retained_source_time','retained_source_time']
    assert projected_motion_review(frames,0,20)['status']=='not_enough_recent_observations'
