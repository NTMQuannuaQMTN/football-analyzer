import numpy as np

from cv.visible_anchor_agreement import visible_foot_disagreement


def test_anchor_comparison_uses_current_image_feet_and_explicit_pixel_units():
    boxes=np.array([[10,20,30,60,.8],[40,20,60,60,.7],[70,20,90,60,.6],[100,20,120,60,.5]])
    h=np.diag([105,68,1.]);anchor=h.copy();anchor[0,2]=12
    assert np.isclose(visible_foot_disagreement(h,anchor,boxes,(1280,720)),12)
    resized=boxes.copy();resized[:,:4]*=1.5
    assert np.isclose(visible_foot_disagreement(h,anchor,resized,(1920,1080)),12)


def test_insufficient_detected_people_do_not_invent_camera_agreement():
    boxes=np.array([[10,20,30,60,.8],[40,20,60,60,.7],[70,20,90,60,.6],[100,20,120,60,.2]])
    assert visible_foot_disagreement(np.eye(3),np.eye(3),boxes,(1280,720)) is None
