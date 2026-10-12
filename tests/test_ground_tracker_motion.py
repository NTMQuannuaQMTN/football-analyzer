import cv2
import numpy as np

from cv.ground_tracker_motion import affine_from_ground


def test_ground_affine_maps_previous_pixels_into_current_pixels():
    image=np.full((360,640,3),(35,115,45),np.uint8)
    for x in range(60,601,60):cv2.line(image,(x,35),(x,335),(235,235,235),2)
    for y in range(60,331,45):cv2.line(image,(35,y),(620,y),(235,235,235),2)
    actual=np.array([[1.,0,18],[0,1.,6]])
    moved=cv2.warpAffine(image,actual,(640,360),borderValue=(35,115,45))
    estimated,diagnostic=affine_from_ground(image,moved)
    assert diagnostic['status']=='accepted'
    points=np.array([[150,100,1],[400,240,1]])
    assert np.linalg.norm(points@estimated.T-points@actual.T,axis=1).max()<1


def test_featureless_ground_retains_identity_warp_without_invented_motion():
    image=np.full((360,640,3),(35,115,45),np.uint8)
    warp,diagnostic=affine_from_ground(image,image)
    assert diagnostic['status']=='unsupported'
    assert np.array_equal(warp,np.eye(2,3))
