from cv.broadcast_views import samples,shot_split,view_kind


def test_closeups_and_alternate_set_piece_angles_are_not_sideline_views():
    assert view_kind({'label':'Main camera center'})=='sideline'
    assert view_kind({'label':'Main behind the goal'})=='alternate'
    assert view_kind({'label':'Close-up corner'})=='closeup'
    assert view_kind({'label':'Close-up player or field referee'})=='closeup'


def test_manual_shots_are_withheld_and_temporal_splits_keep_whole_shots():
    assert shot_split(1400,1480,[1444.2])=='review'
    assert shot_split(650,680,[])=='val'
    assert shot_split(1250,1280,[])=='test'
    assert shot_split(950,980,[])=='train'


def test_replays_and_boundary_samples_never_enter_training_data():
    segments=[(280,330,dict(label='Main camera center',replay='real-time')),
              (330,360,dict(label='Main camera center',replay='replay')),
              (360,390,dict(label='Close-up corner',replay='real-time'))]
    rows=samples(segments,[])
    assert rows and not any(330<=r['video_s']<360 for r in rows)
    assert not any(285<r['video_s']<315 for r in rows)
    assert len({r['part'] for r in rows if r['shot']==0})==1
    assert all(r['target']==0 for r in rows if r['shot']==2)
