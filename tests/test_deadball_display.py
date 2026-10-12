import copy
from backend.analytics.deadball import dead_ball_frames

def event(t,kind,source='soccernet_label'):
    return {'timestamp':t,'event_type':kind,'source':source}

def test_restart_interval_is_half_open_retrospective_and_never_moves_ball_or_players():
    frames=[{'t':t,'b':[t,20],'xy':[[t,30]],'obs':[True]} for t in range(6)]
    before=copy.deepcopy(frames)
    result=dead_ball_frames(frames,[event(1,'FOUL'),event(4,'PENALTY')])
    assert [f.get('ds') for f in result]==[None,'Penalty','Penalty','Penalty',None,None]
    assert result[1]['ds_restart_t']==4 and result[1]['ds_basis']=='annotation_interval'
    assert frames==before
    assert [f['b'] for f in result]==[f['b'] for f in before]
    assert [f['xy'] for f in result]==[f['xy'] for f in before]

def test_sparse_restart_does_not_invent_wait_and_shot_ends_stop_pairing():
    frames=[{'t':t} for t in range(6)]
    assert not any(f.get('ds') for f in dead_ball_frames(frames,[event(4,'CORNER')]))
    assert not any(f.get('ds') for f in dead_ball_frames(frames,[event(1,'FOUL'),event(2,'SHOT'),event(4,'FREE_KICK')]))
    assert not any(f.get('ds') for f in dead_ball_frames(frames,[event(1,'FOUL','estimated'),event(4,'FREE_KICK','estimated')]))

def test_dfl_state_preserved_and_missing_restart_is_explicit_estimate():
    frames=[{'t':0,'ds':'Corner'},{'t':1,'bs':'dead_ball'},{'t':2,'bs':'goal'},{'t':3,'bs':'kick_off'}]
    result=dead_ball_frames(frames,[])
    assert result[0]==frames[0]
    assert result[1]['ds']=='Dead ball' and result[1]['ds_basis']=='estimated_ball_state'
    assert result[2]['ds_basis']=='estimated_ball_state'
    assert 'ds' not in result[3]

def test_served_replay_frames_get_stoppage_fields_and_live_does_not(monkeypatch):
    from types import SimpleNamespace
    import backend.pipeline as pipeline
    frames=[{'t':t,'xy':[[10,20]],'obs':[True],'b':[10,20]} for t in range(5)]
    I=pipeline.MatchIntelligence.__new__(pipeline.MatchIntelligence)
    I.m=SimpleNamespace(meta={'source':{'type':'computer_vision'}},frames=frames,
        events=[event(1,'FOUL'),event(4,'FREE_KICK')],order=['A1'],is_gk=[False])
    I.view='cv';I.alt_frames={'live':copy.deepcopy(frames)};I.recon={};I._frames_cache={};I.timeline={}
    monkeypatch.setattr(pipeline,'lines_per_second',lambda frames,*args,**kwargs:[[0] for f in frames])
    assert I.frames('replay')[1]['ds']=='Free kick'
    assert 'ds' not in I.frames('replay')[4]
    assert not any('ds' in f for f in I.frames('live'))
    assert not any('ds' in f for f in frames)
