import cv2
import numpy as np

from cv.broadcast_motion import BroadcastMotion, association_cost


def test_motion_translation_and_reset_without_future_or_labels():
    frame=np.full((360,640,3),(35,105,40),np.uint8)
    rng=np.random.default_rng(12)
    for x,y in rng.integers([35,35],[605,325],(150,2)):
        cv2.circle(frame,(int(x),int(y)),3,(230,230,230),-1)
    moved=cv2.warpAffine(frame,np.array([[1,0,8],[0,1,4]],float),(640,360),borderValue=(35,105,40))
    motion=BroadcastMotion()
    assert np.allclose(motion.apply(frame),np.eye(2,3))
    warp=motion.apply(moved)
    assert motion.last_diagnostic['accepted']
    assert np.allclose(warp[:,2],[8,4],atol=1)
    motion.reset_params()
    assert np.allclose(motion.apply(moved),np.eye(2,3))


def test_featureless_camera_abstains():
    motion=BroadcastMotion();image=np.zeros((360,640,3),np.uint8)
    assert np.allclose(motion.apply(image),np.eye(2,3))
    assert np.allclose(motion.apply(image),np.eye(2,3))
    assert not motion.last_diagnostic['accepted']


def test_association_guard_rejects_wrong_translation_and_handles_missing_boxes():
    boxes=np.array([[i*40,100,i*40+20,150] for i in range(8)],float)
    moved=boxes+[6,2,6,2]
    correct=np.array([[1,0,6],[0,1,2]],float)
    wrong=np.array([[1,0,-45],[0,1,0]],float)
    assert association_cost(boxes,moved,correct)<1e-6
    assert association_cost(boxes,moved,np.eye(2,3))<association_cost(boxes,moved,wrong)
    assert np.isinf(association_cost([],moved,correct))
