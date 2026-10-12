from pathlib import Path

import pytest

from backend.reconstruction import datasets


def test_camera_sampling_cache_is_specific_and_invalid_rates_are_rejected(monkeypatch,tmp_path):
    monkeypatch.setattr(datasets,'CACHE',tmp_path)
    assert datasets.skillcorner_camera_cache_path(1).name=='skillcorner_cameras.npz'
    assert datasets.skillcorner_camera_cache_path(5).name=='skillcorner_cameras_5hz.npz'
    for rate in [0,-1,3,4,True,2.5]:
        with pytest.raises(ValueError):datasets.skillcorner_camera_cache_path(rate)


def test_consumed_dataset_and_camera_mutations_invalidate_feature_inputs(monkeypatch,tmp_path):
    monkeypatch.setattr(datasets,'CACHE',tmp_path)
    monkeypatch.setattr(datasets,'parse_dfl',lambda ignored: None)
    monkeypatch.setattr(datasets,'skillcorner_cameras',lambda: None)
    match=datasets.dfl_cache_path('test',1);match.write_bytes(b'first arrays')
    camera=datasets.skillcorner_camera_cache_path(1);camera.write_bytes(b'camera arrays')
    first=datasets.reconstruction_input_fingerprints(['test'],['sc_camera'])
    match.write_bytes(b'other arrays')
    second=datasets.reconstruction_input_fingerprints(['test'],['sc_camera'])
    assert first[match.name]!=second[match.name] and first[camera.name]==second[camera.name]
    camera.write_bytes(b'other camera')
    third=datasets.reconstruction_input_fingerprints(['test'],['sc_camera'])
    assert second[camera.name]!=third[camera.name]
    assert camera.name not in datasets.reconstruction_input_fingerprints(['test'],['k=6'])
