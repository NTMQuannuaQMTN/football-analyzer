import pytest

from cv.audit_hidden_sensitivity import require_held_out


def test_stress_evaluation_rejects_training_match_or_missing_split_provenance():
    require_held_out({'train':['other'],'test':'held'},'held')
    for metadata in [{'train':['held'],'test':'held'},{'train':['other'],'test':'different'},{'test':'held'},{}]:
        with pytest.raises(ValueError):require_held_out(metadata,'held')
