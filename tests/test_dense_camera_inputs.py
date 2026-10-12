from copy import deepcopy

import numpy as np
import pytest

from cv.audit_visible_pipeline import dense_camera_cases
from cv.gsr import project


def sample():
    measured={'camera_supported':True,'H':np.diag([105.,68.,1.]).tolist()}
    return {'part':'val','sampling_hz':5,'camera_H_coordinates':'normalised_0_to_1_image_to_pitch_metres','cases':[{'clip':'clip','frame':'000001','methods':{'strict':measured,'foreground_exclusion':deepcopy(measured)}}]}


def test_dense_camera_loader_preserves_normalized_pitch_transform_and_rejects_wrong_cadence_or_split():
    data=sample();cases=dense_camera_cases(data,'val')
    assert np.allclose(project(np.array(cases[0]['H']),[[.5,.5]])[0],[52.5,34.])
    with pytest.raises(ValueError):dense_camera_cases(data,'test')
    data['sampling_hz']=1
    with pytest.raises(ValueError):dense_camera_cases(data,'val')


def test_dense_loader_rejects_duplicated_frames_and_invalid_or_contradictory_cameras():
    data=sample();data['cases']*=2
    with pytest.raises(ValueError,match='Duplicate'):dense_camera_cases(data,'val')
    for value in [{'camera_supported':True,'H':None},{'camera_supported':True,'H':np.zeros((3,3)).tolist()},{'camera_supported':False,'H':np.eye(3).tolist()}]:
        data=sample();data['cases'][0]['methods']['strict']=value
        with pytest.raises(ValueError):dense_camera_cases(data,'val')
