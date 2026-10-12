"""Read permitted local raw timestamps; do not rewrite tracking or model caches."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from backend.reconstruction.datasets import DFL_DIR,HZ,dfl_matches,parse_dfl
from cv.session_report import table


def main():
    root=Path('cv/out/audit8_20261011/dfl_input_audit');root.mkdir(exist_ok=True)
    results=[]
    for match in dfl_matches():
        path=next(DFL_DIR.glob(f'DFL_04_03_positions_raw_observed_*_{match}.xml'))
        before=path.stat();ball={};team=section=None
        for event,element in ET.iterparse(path,events=('start','end')):
            if event=='start' and element.tag=='FrameSet':team=element.get('TeamId');section=element.get('GameSection')
            elif event=='end' and element.tag=='Frame':
                if team=='BALL' and int(element.get('N'))%HZ==0:
                    ball.setdefault(section,[]).append((int(element.get('N')),element.get('T')))
                element.clear()
            elif event=='end' and element.tag=='FrameSet':element.clear()
        after=path.stat()
        if before.st_size!=after.st_size or before.st_mtime_ns!=after.st_mtime_ns:raise RuntimeError('Raw source changed during timing audit')
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
        _,segments=parse_dfl(match);by_section={s['section']:s for s in segments}
        for section,records in sorted(ball.items()):
            frames=[r[0] for r in records];unique=sorted(set(frames));dt=np.diff(unique)/HZ
            timestamps=np.asarray([r[1] for r in sorted({n:t for n,t in records}.items())],dtype='datetime64[ms]')
            utc_dt=np.diff(timestamps).astype('timedelta64[ms]').astype(float)/1000
            segment=by_section[section]
            row=dict(match=match,section=section,retained_raw_ball_frames=len(unique),cached_samples=len(segment['pos']),
                duplicate_retained_ns=len(frames)-len(unique),first_n=unique[0],last_n=unique[-1],first_utc=str(timestamps[0]),last_utc=str(timestamps[-1]),
                frame_number_step_not_1s=int((np.abs(dt-1)>1e-6).sum()),utc_step_not_1s=int((np.abs(utc_dt-1)>.001).sum()),
                max_frame_number_gap_s=float(dt.max()),max_utc_gap_s=float(utc_dt.max()),
                raw_sha256=digest.hexdigest(),cache_count_matches_raw=len(unique)==len(segment['pos']))
            results.append(row)
        (root/'raw_timing.json').write_text(json.dumps(dict(rows=results,conditions='Raw retained BALL frame numbers and UTC timestamps versus legacy 1Hz cache counts. This does not align DFL with broadcast footage or match event clocks. No coordinate/cache/model rewrite.'),indent=2))
        print(match,results[-2:],flush=True)
    rows=[[r['match'],r['section'],r['retained_raw_ball_frames'],r['cached_samples'],r['first_utc'],r['last_utc'],r['frame_number_step_not_1s'],r['utc_step_not_1s'],r['max_utc_gap_s'],r['duplicate_retained_ns']] for r in results]
    (root/'review.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>DFL raw timestamp audit</title><style>body{font:15px system-ui;margin:30px}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:7px}</style><h1>DFL raw timestamp audit</h1><p>Actual local raw frame numbers and UTC times. These verify sampling intervals and cache lengths, not broadcast/event-clock alignment. Source hashes are retained; no raw data or model cache was overwritten.</p>'+table(['Match','Half','Raw 1Hz samples','Cached samples','First UTC','Last UTC','Frame-step gaps','UTC-step gaps','Maximum UTC gap s','Duplicate N'],rows)+'</html>')


if __name__=='__main__':main()
