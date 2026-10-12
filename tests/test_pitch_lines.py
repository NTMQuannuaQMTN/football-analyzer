import numpy as np

from cv.pitch_lines import LINES, homography_from_lines


def test_line_solver_recovers_known_projective_camera():
    world_to_image = np.array([[.008,.002,.05],[.001,.009,.08],[.0007,.001,1]])
    points = {}
    for name in ["Side line top","Side line bottom","Side line left","Side line right"]:
        a,b,c = LINES[name]
        world = np.array([[0,-c],[105,-c]]) if b else np.array([[-c,0],[-c,68]])
        hp = np.column_stack([world,np.ones(2)]) @ world_to_image.T
        points[name] = hp[:,:2]/hp[:,2:]
    h=homography_from_lines(points)
    assert h is not None
    world=np.array([[10,20],[70,50],[52.5,34]])
    hp=np.column_stack([world,np.ones(3)]) @ world_to_image.T
    image=hp[:,:2]/hp[:,2:]
    projected=np.column_stack([image,np.ones(3)]) @ h.T
    assert np.allclose(projected[:,:2]/projected[:,2:],world,atol=1e-6)


def test_insufficient_lines_abstain():
    assert homography_from_lines({"Middle line": [[.4,.1],[.6,.9]], "Side line top": [[0,.1],[1,.1]]}) is None


def test_noisy_three_parallel_lines_do_not_fake_full_rank():
    points={
        'Side line left': [[.1,.2],[.2,.9]],
        'Small rect. left main': [[.2,.21],[.31,.89]],
        'Big rect. left main': [[.4,.19],[.51,.91]],
        'Side line top': [[.1,.2],[.9,.19]],
    }
    assert homography_from_lines(points) is None
