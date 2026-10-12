from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Lock
import time

import pytest

from backend import pipeline


def test_concurrent_cold_requests_share_analysis_and_canonical_paths(monkeypatch,tmp_path):
    source=tmp_path/'match.json';source.write_text('{}')
    calls=[];guard=Lock();barrier=Barrier(6)
    def construct(seed,path):
        with guard:calls.append(path)
        time.sleep(.02)
        return object()
    monkeypatch.setattr(pipeline,'MatchIntelligence',construct)
    monkeypatch.setattr(pipeline,'_cache',{})
    def load(_):
        barrier.wait()
        return pipeline.get_intel(source=source)
    with ThreadPoolExecutor(max_workers=6) as pool:results=list(pool.map(load,range(6)))
    assert len(calls)==1 and all(value is results[0] for value in results)
    assert pipeline.get_intel(source=str(source.parent/'..'/source.parent.name/source.name)) is results[0]


def test_updated_file_invalidates_analysis_without_detector_or_stale_reuse(monkeypatch,tmp_path):
    source=tmp_path/'match.json';source.write_text('first')
    monkeypatch.setattr(pipeline,'_cache',{})
    monkeypatch.setattr(pipeline,'MatchIntelligence',lambda seed,path:Path(path).read_text())
    assert pipeline.get_intel(source=source)=='first'
    source.write_text('replacement')
    assert pipeline.get_intel(source=source)=='replacement'
    assert len(pipeline._cache)==1


def test_failed_construction_is_not_cached(monkeypatch,tmp_path):
    source=tmp_path/'match.json';source.write_text('{}');attempts=[]
    monkeypatch.setattr(pipeline,'_cache',{})
    def construct(seed,path):
        attempts.append(path)
        if len(attempts)==1:raise ValueError('Invalid export')
        return object()
    monkeypatch.setattr(pipeline,'MatchIntelligence',construct)
    with pytest.raises(ValueError,match='Invalid export'):pipeline.get_intel(source=source)
    assert pipeline.get_intel(source=source) is pipeline.get_intel(source=source)
    assert len(attempts)==2
