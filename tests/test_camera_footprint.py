import numpy as np

from cv.clip_extras import footprint
from cv.clip_extras import cached_camera_matrices
import json


def test_camera_footprint_is_exact_and_homogeneous_scale_invariant():
    h=np.array([[.1,0,30],[0,.1,-2],[0,0,1.]])
    polygon=np.asarray(footprint(h))
    assert np.allclose(polygon,np.asarray(footprint(-h)))
    assert np.allclose(polygon.min(0),[30,0])
    assert np.allclose(polygon.max(0),[105,68])
    projected=np.column_stack([polygon,np.ones(len(polygon))]) @ np.linalg.inv(h).T
    xy=projected[:,:2]/projected[:,2:]
    assert (xy>=[-1e-6,-1e-6]).all() and (xy<=[1280+1e-6,720+1e-6]).all()


def test_candidate_footprint_uses_the_projection_camera_and_preserves_abstentions(tmp_path):
    (tmp_path/'run.json').write_text(json.dumps({'camera_H_coordinates':'normalised_0_to_1_image_to_pitch_metres'}))
    h=np.array([[128.,0,30],[0,72.,-2],[0,0,1.]])
    (tmp_path/'detections.jsonl').write_text(json.dumps({'t':1444,'H':h.tolist()})+'\n'+json.dumps({'t':1445,'H':None})+'\n')
    cameras=cached_camera_matrices(tmp_path)
    assert np.allclose(cameras[1444],[[.1,0,30],[0,.1,-2],[0,0,1.]])
    assert cameras[1445] is None
    assert np.allclose(np.asarray(footprint(cameras[1444])).min(0),[30,0])
