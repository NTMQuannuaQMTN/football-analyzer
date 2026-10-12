import numpy as np

from cv.appearance_tracker import appearance_costs


def test_appearance_only_links_unique_nearby_supported_matches():
    base=np.array([[.5,.4],[.4,.5]])
    assert np.allclose(appearance_costs(base,np.full((2,2),.95),np.ones((2,2),bool)),base)
    result=appearance_costs(base,np.array([[.98,.91],[.91,.98]]),np.ones((2,2),bool))
    assert result[0,0]<.02 and result[1,1]<.02
    assert result[0,1]==base[0,1]
    assert np.array_equal(appearance_costs(base,np.array([[.98,.91],[.91,.98]]),np.zeros((2,2),bool)),base)
    assert np.array_equal(appearance_costs(base,np.full((2,2),np.nan),np.ones((2,2),bool)),base)
