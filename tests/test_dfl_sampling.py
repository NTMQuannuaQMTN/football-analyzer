import numpy as np
import pytest

from backend.reconstruction.datasets import dfl_cache_path,initial_home_orientation,rows
from backend.reconstruction.real_benchmark import segment_view


def test_sampling_rate_cannot_reuse_one_hz_cache_or_silently_floor_to_other_rate():
    assert dfl_cache_path('match',1)!=dfl_cache_path('match',5)
    assert dfl_cache_path('match',5)!=dfl_cache_path('match',25)
    for rate in [0,2,10,1.5,True]:
        with pytest.raises(ValueError):dfl_cache_path('match',rate)


def test_initial_orientation_ignores_future_goalkeeper_positions():
    pos=np.array([[[5.,34.]],[[95.,34.]],[[95.,34.]]])
    assert not initial_home_orientation(pos,['A'],['GK'])
    assert not initial_home_orientation(pos[:1],['A'],['GK'])
    pos[0,0,0]=95
    assert initial_home_orientation(pos,['A'],['GK'])
    pos[0]=np.nan
    with pytest.raises(ValueError):initial_home_orientation(pos,['A'],['GK'])


def test_unified_rows_use_recorded_sample_times_instead_of_frame_index():
    seg={'pos':np.array([[[5.,6.]],[[7.,8.]]]),'section':'firstHalf','pids':['p'],'team':['A'],'role':['GK'],'sampling_hz':5,'timestamps_s':[0,.2]}
    records=list(rows({'match_id':'m'},seg))
    assert [r['timestamp'] for r in records]==[0.,.2]
    del seg['timestamps_s']
    assert [r['timestamp'] for r in rows({'match_id':'m'},seg)]==[0.,.2]


def test_one_hz_model_rejects_faster_or_gapped_tracking():
    for segment in [{'sampling_hz':5},{'sampling_hz':1,'timestamps_s':[0,1,3]}]:
        with pytest.raises(ValueError,match='uniform 1 Hz'):segment_view({},segment)
