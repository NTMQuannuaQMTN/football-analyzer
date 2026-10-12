from cv.cache_provenance import files_digest


def test_cached_experiment_identity_changes_when_input_content_changes(tmp_path):
    a=tmp_path/'image';b=tmp_path/'detector'
    a.write_bytes(b'image');b.write_bytes(b'boxes')
    original=files_digest([a,b])
    assert files_digest([b,a,a])==original
    b.write_bytes(b'other boxes')
    assert files_digest([a,b])!=original
