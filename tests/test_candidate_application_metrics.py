import copy
import pytest
from cv.apply_seven_hour_candidates import position_metrics,candidate_roles


def sample():
    return {'frames':[{'xy':[[0,0],None]},{'xy':[[1,0],[10,10]]}],
      'alt_frames':{'live':[{'xy':[[0,0],None],'obs':[True,None],'rad':[0,None]},
                           {'xy':[[2,0],[11,10]],'obs':[False,False],'rad':[2,2]}]}}


def test_substitute_nulls_do_not_become_zero_positions_or_contaminate_errors():
    data=sample();before=copy.deepcopy(data);r=position_metrics(data,'live')
    assert r['all']['n']==3 and r['hidden']['n']==2
    assert r['hidden']['mean_m']==1
    assert r['last_observed_paired']['n']==1
    assert r['by_observation_age']['Never observed']['n']==1
    assert r['uncertainty_coverage']['inside_radius_fraction']==1
    assert data==before


def test_mutated_observed_ground_truth_cannot_be_claimed_as_camera_accuracy():
    data=sample();data['alt_frames']['live'][0]['xy'][0]=[5,0]
    with pytest.raises(AssertionError,match='mutated'):position_metrics(data,'live')
    baseline=position_metrics(data,'live',require_preserved_observations=False)
    assert not baseline['observed_coordinates_preserved']
    assert baseline['observed']['mean_m']==5


def test_skillcorner_position_names_are_not_parsed_as_numeric_formation_slots():
    result=candidate_roles([{'id':'A1','role_slot':'GK'},{'id':'A2','role_slot':'RWB'},
        {'id':'A3','role_slot':'CF'},{'id':'B2','role_slot':'LCB'}])
    assert result=={'A1':'GK','A2':'FB','A3':'ST','B2':'CB'}
    with pytest.raises(KeyError):candidate_roles([{'id':'A1','role_slot':'unknown'}])
