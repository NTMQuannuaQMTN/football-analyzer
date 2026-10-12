from types import SimpleNamespace

import pytest

from backend.pipeline import MatchIntelligence


def video_intelligence():
    I=MatchIntelligence.__new__(MatchIntelligence)
    I.m=SimpleNamespace(meta={'source':{'type':'computer_vision'}})
    I.view='cv';I.alt_frames={}
    return I


def test_retrospective_video_cannot_be_served_as_live_or_truth():
    I=video_intelligence()
    assert I.resolve_mode('default')=='replay'
    for mode in ('live','truth'):
        with pytest.raises(ValueError,match='retrospective'):
            I.frames(mode)
        with pytest.raises(ValueError,match='retrospective'):
            I.player_states(0,mode)
        with pytest.raises(ValueError,match='retrospective'):
            I.player_state('A14',0,mode)


def test_explicit_causal_export_and_tracking_modes_remain_supported():
    I=video_intelligence();I.alt_frames={'live':[]}
    assert I.resolve_mode('live')=='live'
    I.m.meta={'source':{'type':'tracking'}}
    assert I.resolve_mode('truth')=='truth'
    with pytest.raises(ValueError,match='Unknown'):
        I.resolve_mode('typo')


def test_live_player_state_and_history_do_not_reveal_hidden_tracking_truth():
    I=video_intelligence();I.view='truth';I.recon={};I._frames_cache={}
    I.timeline={'A':[],'B':[]}
    I.m=SimpleNamespace(meta={'source':{'type':'tracking'}},duration=1,order=['A14'],idx={'A14':0},is_gk=[False],
        frames=[{'t':t,'xy':[[99,60]],'obs':[True]} for t in range(2)])
    I.alt_frames={'live':[{'t':t,'xy':[[10+t,20]],'obs':[False],'conf':[.2],'rad':[15]} for t in range(2)]}
    state=I.player_state('A14',0,'live')
    assert state['position']=={'x':10,'y':20}
    assert state['state_source']=='inferred'
    assert state['history']['last_seen_position'] is None
    assert state['history']['last_seen_t'] is None
    assert 'next_seen_position' not in state['history']


def test_video_tracking_confidence_is_not_presented_as_video_probability():
    I=video_intelligence();I.recon={};I.m.duration=0;I.m.order=['A14','A8']
    I.m.frames=[{'xy':[[10,20],[20,30]],'obs':[True,False],'conf':[1,.95],'rad':[0,12]}]
    observed,inferred=I.player_states(0)
    assert observed['confidence'] is None and observed['uncertainty_radius'] is None
    assert inferred['confidence'] is None and inferred['uncertainty_radius']==12


def test_video_measurement_timestamp_is_preserved_without_inferred_timestamp():
    I=video_intelligence();I.recon={};I.m.duration=0;I.m.order=['A14','A8']
    I.m.frames=[{'xy':[[10,20],[20,30]],'obs':[True,False],'observed_t':[.4,None]}]
    observed,inferred=I.player_states(0)
    assert observed['observation_timestamp']==.4
    assert inferred['observation_timestamp'] is None


def test_live_line_labels_ignore_future_frames_and_truth_timeline():
    import copy
    from backend.analytics.shape import lines_per_second
    order=[f'A{i}' for i in range(10)]
    xy=[[20+(i//4)*20,8+(i%4)*16] for i in range(10)]
    frames=[{'xy':copy.deepcopy(xy)} for _ in range(65)]
    changed=copy.deepcopy(frames)
    for fr in changed[10:]:
        fr['xy']=[[100-x,68-y] for x,y in fr['xy']]
    timeline={'A':[{'formation':'4-3-3'}]*2,'B':[]}
    changed_timeline={'A':[{'formation':'5-4-1'}]*2,'B':[]}
    first=lines_per_second(frames,order,[False]*10,timeline,causal=True)
    second=lines_per_second(changed,order,[False]*10,changed_timeline,causal=True)
    assert first[:10]==second[:10]
