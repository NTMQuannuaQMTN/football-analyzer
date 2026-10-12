"""Review projected observation contradictions without repairing coordinates.

This diagnoses a role-assigned export, not verified physical player speed or
position confidence. It never uses inferred points or future timestamps.
"""
import math


def projected_motion_review(frames,player_index,t,max_gap_s=5.,max_age_s=10.,review_speed_m_s=10.):
    measured=[]
    for index,frame in enumerate(frames):
        label_time=frame.get('t',index)
        if label_time>t:break
        if (frame.get('obs') or [None]*len(frame['xy']))[player_index] is not True:continue
        point=frame['xy'][player_index]
        if point is None or len(point)!=2 or not all(math.isfinite(v) for v in point):continue
        source_time=(frame.get('observed_t') or [None]*len(frame['xy']))[player_index]
        time=label_time if source_time is None else source_time
        if not math.isfinite(time) or time>t:continue
        measured.append((time,point,'frame_label_time' if source_time is None else 'retained_source_time'))
    measured.sort(key=lambda item:item[0])
    result=dict(status='not_enough_recent_observations',apparent_speed_m_s=None,
        review_threshold_m_s=review_speed_m_s,
        basis='Two recent projected observations with role-assigned identity. A contradiction can be camera, identity, timing or detection error; it is not verified player speed or a calibrated accuracy probability.')
    if len(measured)<2:return result
    previous,current=measured[-2:];dt=current[0]-previous[0]
    if not 0<dt<=max_gap_s or t-current[0]>max_age_s:return result
    distance=math.dist(previous[1],current[1]);speed=distance/dt
    result.update(status='review_camera_or_identity' if speed>review_speed_m_s else 'no_large_recent_jump',
        apparent_speed_m_s=round(speed,2),displacement_m=round(distance,2),elapsed_s=round(dt,3),
        previous_t=previous[0],current_t=current[0],timing_basis=[previous[2],current[2]],
        limitation='A small apparent jump does not verify the camera, named identity or position. Coordinates were not changed.')
    return result
