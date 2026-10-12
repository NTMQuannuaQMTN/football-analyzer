from cv.audit_camera_uncertainty import fit_bins,evaluate


def test_camera_coverage_is_scored_on_test_not_used_to_refit_bins():
    calibration=fit_bins([dict(bin='anchor_small',errors_m=[1.]*80+[10.]*20)])
    radius=calibration['anchor_small']['radius_80_m']
    result=evaluate([dict(bin='anchor_small',errors_m=[1.,20.]),dict(bin='unknown',errors_m=[2.])],calibration)
    assert result['n']==2 and result['radius_coverage']==.5
    assert result['unsupported_calibration_people']==1
    assert calibration['anchor_small']['radius_80_m']==radius
