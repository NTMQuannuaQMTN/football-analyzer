import json

import numpy as np
import pytest

from cv.evaluate_manual_camera import cached_camera


def test_candidate_camera_units_and_exact_timestamp_are_required(tmp_path):
    (tmp_path/'run.json').write_text(json.dumps({'camera_H_coordinates':'normalised_0_to_1_image_to_pitch_metres'}))
    (tmp_path/'detections.jsonl').write_text(json.dumps({'t':1800.2,'H':[[105,0,0],[0,68,0],[0,0,1]]})+'\n')
    camera=cached_camera(tmp_path)
    np.testing.assert_allclose(camera(1800.2)@np.array([640,360,1]),[52.5,34,1])
    assert camera(1800.) is None  # Never silently choose another frame.
    (tmp_path/'run.json').write_text('{}')
    with pytest.raises(ValueError,match='coordinates'):cached_camera(tmp_path)


def test_duplicate_candidate_timestamps_are_rejected(tmp_path):
    (tmp_path/'run.json').write_text(json.dumps({'camera_H_coordinates':'normalised_0_to_1_image_to_pitch_metres'}))
    line=json.dumps({'t':1800.,'H':None})+'\n'
    (tmp_path/'detections.jsonl').write_text(line+line)
    with pytest.raises(ValueError,match='Duplicate'):cached_camera(tmp_path)
