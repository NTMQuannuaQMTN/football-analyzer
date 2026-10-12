import copy

import numpy as np

from cv.clean_line_labels import clean_row
from cv.gsr import project


def test_training_label_cleanup_corrects_semantics_without_mutating_positions():
    camera=np.array([[9,2,100],[1,7,120],[.001,.002,1.]])
    world=np.array([[x,y] for x in [10,35,65,95] for y in [12,34,56]])
    feet=project(camera,world)
    people=[dict(pitch=p.tolist(),role='player',box=[f[0]-5,f[1]-25,f[0]+5,f[1]]) for p,f in zip(world,feet)]
    line=project(camera,np.array([[105,0],[105,68]]))/[1920,1080]
    row=dict(people=people,lines={'Big rect. left top':[dict(x=x,y=y) for x,y in line]})
    original=copy.deepcopy(row)
    clean,d=clean_row(row)
    assert d['trusted'] and d['changes'][0]['corrected']=='Side line right'
    assert 'Side line right' in clean['lines']
    assert clean['people']==original['people'] and row==original


def test_untrusted_training_frame_retains_labels():
    row=dict(people=[],lines={'Middle line':[dict(x=.5,y=.1),dict(x=.5,y=.9)]})
    result,d=clean_row(row)
    assert result==row and not d['trusted']
