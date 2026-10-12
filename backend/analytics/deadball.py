"""Retrospective stoppage display from explicit event annotations, without moving coordinates."""
import math

RESTARTS={'FREE_KICK':'Free kick','PENALTY':'Penalty','CORNER':'Corner',
          'THROW_IN':'Throw-in','KICK_OFF':'Kick-off','GOAL_KICK':'Goal kick'}
STOPS={'FOUL','BALL_OUT','OFFSIDE','GOAL'}


def dead_ball_frames(frames,events):
    # Source annotations are sparse. Pair only explicit stops with a restart;
    # a restart alone does not establish when the ball became dead.
    annotations=sorted((e for e in events if e.get('source')=='soccernet_label'),key=lambda e:e['timestamp'])
    intervals=[]
    for i,e in enumerate(annotations):
        if e['event_type'] not in STOPS:continue
        for following in annotations[i+1:]:
            kind=following['event_type']
            if kind in RESTARTS:
                if following['timestamp']>e['timestamp']:
                    intervals.append((e['timestamp'],following['timestamp'],RESTARTS[kind]))
                break
            if kind in ('SHOT','GOAL'):
                break  # recorded play resumed before a later unrelated restart
    output=[]
    for frame in frames:
        f=dict(frame)
        if not f.get('ds'):
            interval=next((x for x in reversed(intervals) if x[0]<=frame['t']<x[1]),None)
            if interval:
                f.update(ds=interval[2],ds_basis='annotation_interval',ds_restart_t=interval[1])
            elif frame.get('bs') in ('dead_ball','goal'):
                f.update(ds='Dead ball',ds_basis='estimated_ball_state')
        output.append(f)
    return output
