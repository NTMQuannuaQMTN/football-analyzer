import numpy as np
import pytest

from cv.audit_context_comparison import block_comparison


def test_block_comparison_keeps_exact_pairing_and_constant_error_change():
    indices=np.array([[0,0,1],[0,61,1],[1,0,2],[1,61,2]])
    control=np.array([2.,4.,6.,8.]);candidate=control+2
    result=block_comparison(control,indices,candidate,indices,60,draws=100)
    assert result['n']==4 and result['temporal_blocks']==4
    assert result['candidate_minus_control_mean_m']==2
    assert result['interval_95_m']==[2,2]
    with pytest.raises(ValueError,match='identical'):
        block_comparison(control,indices,candidate,indices[::-1],60)
