import numpy as np
import torch

from cv.physical_camera import camera_matrix,project


def test_physical_camera_aim_projects_to_image_centre_and_round_trips_ground():
    pose=torch.tensor([[52.5,95.,22.,80.,34.,1.5,0.]],dtype=torch.float64)
    matrix=camera_matrix(pose);points=torch.tensor([[[80.,34.],[85.,40.],[90.,20.]]],dtype=torch.float64)
    image=project(matrix,points)
    assert torch.allclose(image[0,0],torch.tensor([.5,.5],dtype=torch.float64))
    assert torch.allclose(project(torch.linalg.inv(matrix),image),points,atol=1e-8)
    image.sum().backward() if pose.requires_grad else None
    assert np.isfinite(matrix.numpy()).all()


def test_camera_projection_has_finite_gradients_for_training_correspondences():
    pose=torch.tensor([[52.5,95.,22.,80.,34.,1.5,0.]],requires_grad=True)
    image=project(camera_matrix(pose),torch.tensor([[[75.,25.],[95.,45.]]]))
    image.sum().backward()
    assert torch.isfinite(pose.grad).all() and pose.grad.abs().sum()>0
