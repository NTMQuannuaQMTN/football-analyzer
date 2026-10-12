import numpy as np
import pytest

from cv.temporal_samples import sample_seconds


def test_integer_observation_does_not_average_future_motion():
    rows=[(t,20+5*t,30) for t in [0,.2,.4,.6,.8]]
    mean,times=sample_seconds(rows,1)
    near,nt=sample_seconds(rows,1,'nearest')
    assert np.allclose(mean[0],[22,30]) and times[0]==pytest.approx(.4)
    assert np.allclose(near[0],[20,30]) and nt[0]==0


def test_causal_sampling_never_uses_future_and_preserves_source_timestamp():
    rows=[(.2,10,20),(.8,11,21),(1.2,30,40)]
    positions,times=sample_seconds(rows,2,'nearest',causal=True)
    assert 0 not in positions and times[1]==.8
    assert np.allclose(positions[1],[11,21])
    with pytest.raises(ValueError,match='future'):sample_seconds(rows,2,causal=True)
