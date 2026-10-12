import cv2
import numpy as np

from cv.camera_similarity import similarity_fallback


def test_similarity_recovers_narrow_ground_translation_without_player_labels():
    image=np.full((360,640,3),(35,130,35),np.uint8)
    for x in range(70,370,25):cv2.rectangle(image,(x,170),(x+12,182),(235,235,235),2)
    current=cv2.warpAffine(image,np.array([[1,0,6],[0,1,2]],np.float32),(640,360),borderValue=(35,130,35))
    h,d=similarity_fallback(image,current)
    assert h is not None and d['status']=='measured_similarity'
    p=h@np.array([.5+6/640,.5+2/360,1])
    assert np.linalg.norm(p[:2]/p[2]-[.5,.5])<.003


def test_similarity_abstains_without_ground_support():
    image=np.full((360,640,3),(35,130,35),np.uint8)
    h,d=similarity_fallback(image,image)
    assert h is None and d['status']=='unsupported'
