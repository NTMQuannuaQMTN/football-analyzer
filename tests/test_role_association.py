import numpy as np
import pytest

from cv.role_association import assign


def test_existing_track_alias_does_not_swap_when_formation_prior_changes():
    cost=np.array([[50,1],[1,50]])
    assert assign(cost,[10,20],[10,20],stable=False)==[(0,1),(1,0)]
    assert assign(cost,[10,20],[10,20],stable=True)==[(0,0),(1,1)]


def test_new_slots_remain_one_to_one_and_explicit_identity_can_override_alias():
    cost=np.array([[200,0],[0,200]])
    assert assign(cost,[10,30],[10,None],stable=True,override={10:1})==[(0,1),(1,0)]
    assert assign(np.array([[100,200]]),[30],[10,20],stable=True)==[]
    with pytest.raises(ValueError,match='Duplicate'):assign(np.zeros((2,2)),[10,10],[None,None],stable=True)
