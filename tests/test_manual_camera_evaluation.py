import numpy as np
import pytest

from cv.evaluate_manual_camera import read_labels,score


def test_manual_landmarks_score_unchanged_camera_and_exclude_replays():
    labels=[dict(frame_id='one',video_s=1,kind='landmark',label='corner',pixel_x=3,pixel_y=4,world_x=0,world_y=0),
            dict(frame_id='two',video_s=2,kind='landmark',label='corner',pixel_x=3,pixel_y=4,world_x=0,world_y=0),
            dict(frame_id='one',video_s=1,kind='identity',label='14')]
    h=np.eye(3);result=score(labels,lambda t:h,lambda t:t==1)
    assert result['scored_landmarks']==1 and result['mean_landmark_error_m']==5
    assert result['readable_identity_labels']==1 and result['rows'][1]['error_m'] is None
    assert np.array_equal(h,np.eye(3))


def test_manual_import_rejects_mismatched_timestamps(tmp_path):
    path=tmp_path/'labels.csv'
    path.write_text('frame_id,video_s,kind,label,pixel_x,pixel_y,world_x,world_y,notes\none,2,landmark,corner,3,4,0,0,\n')
    with pytest.raises(ValueError,match='timestamp'):read_labels(path,[dict(id='one',video_s=1)])


def test_fabricated_ui_smoke_clicks_cannot_be_scored_as_human_truth(tmp_path):
    path=tmp_path/'labels.csv'
    path.write_text('frame_id,video_s,kind,label,pixel_x,pixel_y,world_x,world_y,notes\none,1,landmark,corner,3,4,0,0,UI SMOKE TEST ONLY\n')
    with pytest.raises(ValueError,match='not evaluation references'):
        read_labels(path,[dict(id='one',video_s=1)])


def test_assistant_provisional_readings_require_their_declared_basis(tmp_path):
    path=tmp_path/'labels.csv'
    path.write_text('frame_id,video_s,kind,label,pixel_x,pixel_y,world_x,world_y,notes\none,1,landmark,corner,3,4,0,0,Assistant visual reading; Not independent human GT\n')
    frames=[dict(id='one',video_s=1)]
    with pytest.raises(ValueError,match='not human ground truth'):read_labels(path,frames)
    labels=read_labels(path,frames,basis='assistant_provisional')
    assert len(labels)==1 and labels[0]['pixel_x']==3
