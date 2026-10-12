import numpy as np

from cv.camera_geometry import fit_camera, image_line_residuals, consensus_camera
from cv.pitch_lines import LINES
from cv.gsr import project


def test_image_geometry_preserves_exact_camera():
    camera=np.array([[.008,.002,.05],[.001,.009,.08],[.0007,.001,1]])
    points={}
    for name in ['Side line top','Side line bottom','Side line left','Side line right','Middle line']:
        a,b,c=LINES[name]
        world=np.array([[0,-c],[105,-c]]) if b else np.array([[-c,0],[-c,68]])
        points[name]=project(camera,world)
    h,diagnostic=fit_camera(points)
    assert h is not None and diagnostic['supported']
    assert np.max(np.abs(image_line_residuals(h,points)))<1e-7
    world=np.array([[10,20],[70,50]])
    assert np.allclose(project(h,project(camera,world)),world,atol=1e-5)


def test_camera_geometry_rejects_insufficient_constraints():
    h,d=fit_camera({'Middle line':[[.3,0],[.5,1]]})
    assert h is None and not d['supported']


def test_consensus_rejects_misnamed_ground_line_without_player_truth():
    camera=np.array([[.008,.002,.05],[.001,.009,.08],[.0007,.001,1]])
    points={}
    for name in ['Side line top','Side line bottom','Side line left','Side line right','Middle line']:
        a,b,c=LINES[name]
        world=np.array([[0,-c],[105,-c]]) if b else np.array([[-c,0],[-c,68]])
        points[name]=project(camera,world)
    points['Big rect. left top']=project(camera,np.array([[105,0],[105,68]]))
    h,d=consensus_camera(points)
    assert 'Big rect. left top' in d['rejected_lines']
    world=np.array([[10,20],[70,50]])
    assert np.allclose(project(h,project(camera,world)),world,atol=1e-5)
