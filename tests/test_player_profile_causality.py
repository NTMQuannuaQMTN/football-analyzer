from types import SimpleNamespace

from backend.models.match import MatchData
from backend.pipeline import MatchIntelligence


def profile_fixture():
    raw=dict(meta=dict(duration=2,source={'type':'tracking'},player_order=['A1']),
        players=[dict(id='A1',team='A',name='Player',shirt=1,position='MID')],
        events=[dict(timestamp=2,player_id='A1',event_type='GOAL',team_id='A',x=100,y=34)],
        frames=[dict(t=t,xy=[[90,60]],obs=[True],b=[50,34],p='A') for t in range(3)])
    frames=[dict(t=t,xy=[[10+8*t,20]],obs=[t==0],b=[50,34],p='A') for t in range(3)]
    I=MatchIntelligence.__new__(MatchIntelligence);I.m=MatchData(raw);I.view='truth';I.recon={}
    I.alt_frames={'live':frames};I.frames=lambda mode:frames if mode=='live' else I.m.frames
    I.moments=[dict(id='future-derived',timestamp=1,title='Retrospective mechanism',minute_label="1'",replay=dict(focus_players=['A1']))]
    I.gen=SimpleNamespace(player_summary=lambda p,s,mos:dict(headline='LIVE IMPACT',impact='Observed events',moments=[m['title'] for m in mos]))
    return I,frames


def test_live_profile_uses_selected_prefix_without_hidden_truth_or_later_events():
    I,frames=profile_fixture();before=I.player_profile('A1',1,'live')
    assert before['until']==1 and before['mode']=='live'
    assert before['movement']['distance_km']==.01 and before['movement']['top_speed_kmh']==0
    assert before['stats']['goals']==0 and before['moments']==[]
    assert before['visibility']['observed_share']==.5
    frames[2]['xy']=[[999,999]];I.m.frames[0]['xy']=[[1,1]];I.m.xy[:]=999
    I.m.events[0]['timestamp']=1.5
    assert I.player_profile('A1',1,'live')==before


def test_replay_keeps_completed_moment_links_and_profile_time_is_clamped():
    I,_=profile_fixture();result=I.player_profile('A1',100,'replay')
    assert result['until']==2 and result['stats']['goals']==1
    assert len(result['moments'])==1 and result['narrative']['headline']=='PLAYER — REPLAY SUMMARY'
    assert I.player_profile('A1',-5,'live')['until']==0
