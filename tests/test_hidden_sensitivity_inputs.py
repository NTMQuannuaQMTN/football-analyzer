import numpy as np

from cv.audit_hidden_sensitivity import CONDITIONS, corrupt


def segment():
    return dict(pos=np.arange(6*4*2, dtype=float).reshape(6, 4, 2), ball=np.arange(12, dtype=float).reshape(6, 2),
                team=np.array(['A', 'A', 'B', 'B']), role=np.array(['CM']*4))


def test_delayed_measurements_only_use_previous_available_observations():
    sample=segment();visible=np.ones((6,4),bool);visible[0,0]=False
    sample['pos'][0,0]=[999,999]  # Unavailable reference must not become an input.
    measurement,available,_=corrupt(sample,visible,'measurement_delay_1s',11026)
    assert not available[0].any()
    assert not available[1,0] and np.isnan(measurement[1,0]).all()
    np.testing.assert_equal(measurement[1,1],sample['pos'][0,1])


def test_stresses_preserve_inputs_and_do_not_depend_on_future_positions():
    sample=segment();visible=np.ones((6,4),bool);visible[:3,0]=False
    original=sample['pos'].copy()
    changed={**sample,'pos':sample['pos'].copy(),'ball':sample['ball'].copy()}
    changed['pos'][3:]=1000;changed['ball'][3:]=1000
    for condition in CONDITIONS:
        first=corrupt(sample,visible,condition,11026)
        second=corrupt(changed,visible,condition,11026)
        for left,right in zip(first,second):np.testing.assert_equal(left[:3],right[:3])
        assert not np.isfinite(first[0][~first[1]]).any()
    np.testing.assert_equal(sample['pos'],original)
