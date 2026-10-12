from cv.evaluate_manual_identity import associate


def person(track,x):
    return dict(cls='person',id=track,box=[x-5,10,x+5,50])


def test_human_foot_click_links_anonymous_track_without_pitch_coordinates():
    row=dict(live=True,dets=[person(8,40)],unassigned_dets=[])
    result=associate(dict(pixel_x=41,pixel_y=51),row)
    assert result['track']==8 and result['status']=='anonymous track linked'
    assert 'px' not in row['dets'][0]


def test_close_feet_and_missing_detections_are_not_forced_to_an_identity():
    row=dict(live=True,dets=[person(8,40),person(9,44)],unassigned_dets=[])
    label=dict(pixel_x=41,pixel_y=50)
    assert associate(label,row)['track'] is None
    assert associate(dict(pixel_x=100,pixel_y=50),row)['track'] is None
    assert associate(label,None)['track'] is None


def test_unconfirmed_detections_and_replay_are_distinguished_from_tracks():
    row=dict(live=True,dets=[],unassigned_dets=[person(None,40)])
    label=dict(pixel_x=40,pixel_y=50)
    assert associate(label,row)['status']=='detection unconfirmed'
    row['live']=False
    assert associate(label,row)['status']=='N/A'


def test_legacy_unclassified_boxes_do_not_become_invented_player_identities():
    row=dict(live=True,dets=[],unassigned_dets=[dict(id=None,box=[35,10,45,50])])
    assert associate(dict(pixel_x=40,pixel_y=50),row)['status']=='N/A'


def test_body_click_can_link_an_image_box_without_becoming_a_foot_reference():
    row=dict(live=True,dets=[person(8,40)],unassigned_dets=[])
    label=dict(pixel_x=40,pixel_y=15)
    assert associate(label,row)['track'] is None
    result=associate(label,row,point_kind='body')
    assert result['track']==8 and result['status']=='anonymous box linked'
    assert 'distance_pixels' not in result
    assert 'no ground-contact point' in result['reason']


def test_body_click_never_selects_between_overlapping_boxes_or_uses_a_cut():
    row=dict(live=True,dets=[person(8,40),person(9,44)],unassigned_dets=[])
    label=dict(pixel_x=41,pixel_y=15)
    assert associate(label,row,point_kind='body')['track'] is None
    row['live']=False
    assert associate(label,row,point_kind='body')['track'] is None
