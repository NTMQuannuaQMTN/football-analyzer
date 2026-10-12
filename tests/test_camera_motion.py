import cv2
import numpy as np

from cv.camera_motion import ground_motion,propagate


def test_ground_motion_recovers_translation_from_painted_pitch():
    image=np.full((360,640,3),(35,130,35),np.uint8)
    for x in [50,200,350,520]:
        for y in [60,130,210,300]:cv2.rectangle(image,(x,y),(x+40,y+20),(235,235,235),2)
    current=cv2.warpAffine(image,np.array([[1,0,8],[0,1,4]],np.float32),(640,360),borderValue=(35,130,35))
    h,d=ground_motion(image,current)
    assert h is not None and d['inliers']>=16
    point=h@np.array([.5+8/640,.5+4/360,1]);point=point[:2]/point[2]
    assert np.linalg.norm(point-[.5,.5])<.002


def test_propagation_requires_supported_current_and_prior_images():
    image=np.full((360,640,3),(35,130,35),np.uint8)
    h,d=ground_motion(image,image)
    assert h is None and d['status']=='unsupported'
    assert propagate(np.eye(3),h) is None
    h=propagate(np.diag([105,68,1]),np.eye(3))
    p=h@np.array([.5,.5,1]);assert np.allclose(p[:2]/p[2],[52.5,34])
