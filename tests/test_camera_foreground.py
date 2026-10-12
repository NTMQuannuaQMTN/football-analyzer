import cv2
import numpy as np

from cv.camera_foreground import detected_ground_mask,motion_without_people


def test_detected_foreground_is_removed_in_resized_pixel_coordinates():
    image=np.full((360,640,3),(35,130,35),np.uint8)
    cv2.rectangle(image,(90,90),(120,120),(235,235,235),2)
    assert detected_ground_mask(image,[],(1280,720))[90,100]>0
    assert detected_ground_mask(image,[[160,160,260,260]],(1280,720))[90,100]==0


def test_background_translation_is_recovered_after_foreground_exclusion():
    image=np.full((360,640,3),(35,130,35),np.uint8)
    for x in [50,200,350,520]:
        for y in [60,130,210,300]:cv2.rectangle(image,(x,y),(x+40,y+20),(235,235,235),2)
    current=cv2.warpAffine(image,np.array([[1,0,8],[0,1,4]],np.float32),(640,360),borderValue=(35,130,35))
    h,d=motion_without_people(image,current,[[200,130,240,150]],[[208,134,248,154]])
    assert h is not None and d['inliers']>=16
    p=h@np.array([.5+8/640,.5+4/360,1]);assert np.linalg.norm(p[:2]/p[2]-[.5,.5])<.002
