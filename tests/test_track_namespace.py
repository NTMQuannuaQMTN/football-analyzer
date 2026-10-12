import copy

import pytest

from cv.track_namespace import TrackNamespace, qualify_rows


def test_native_id_reuse_after_camera_cut_is_not_identity_continuity():
    rows = [{'t': 0, 'live': True, 'dets': [{'cls': 'person', 'id': 1, 'px': 10, 'py': 20}]},
            {'t': .2, 'live': True, 'dets': [{'cls': 'person', 'id': 1, 'px': 11, 'py': 20}]},
            {'t': .4, 'live': False, 'dets': []},
            {'t': .6, 'live': False, 'dets': []},
            {'t': .8, 'live': True, 'dets': [{'cls': 'person', 'id': 1, 'px': 90, 'py': 30}, {'cls': 'ball', 'id': -1, 'px': 90, 'py': 30}]}]
    preserved = copy.deepcopy(rows)
    result = qualify_rows(rows)
    assert rows == preserved
    assert result[0]['dets'][0]['id'] == result[1]['dets'][0]['id']
    assert result[0]['dets'][0]['id'] != result[-1]['dets'][0]['id']
    assert result[-1]['dets'][0]['native_id'] == 1
    assert result[-1]['dets'][1]['id'] == -1
    assert qualify_rows(rows[:2]) == result[:2]  # Future cuts cannot change past IDs.
    assert qualify_rows(result) == result  # Safe to recognise an already qualified cache.


def test_missing_camera_flags_and_unsupported_measurements_are_explicit():
    with pytest.raises(ValueError, match='flags'):
        qualify_rows([{'t': 0, 'dets': []}])
    namespace = TrackNamespace()
    with pytest.raises(ValueError, match='unsupported'):
        namespace.identity(1)
