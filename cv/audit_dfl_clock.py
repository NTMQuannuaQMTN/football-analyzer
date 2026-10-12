"""Verify actual source-frame continuity behind the legacy 1 Hz tracking caches."""
import glob
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from backend.reconstruction.datasets import DFL_DIR, HZ, _match_info
from backend.reconstruction.real_benchmark import split


def audit(mid):
    info=_match_info(mid)
    keepers={pid for pid,p in info['players'].items() if p['team']=='A' and p['role']=='GK'}
    path=glob.glob(str(DFL_DIR/f'DFL_04_03_positions_raw_observed_*_{mid}.xml'))[0]
    grids={};gx={};section=team=person=None
    for event,element in ET.iterparse(path,events=('start','end')):
        if event=='start' and element.tag=='FrameSet':
            section=element.get('GameSection');team=element.get('TeamId');person=element.get('PersonId')
        elif event=='end' and element.tag=='Frame':
            n=int(element.get('N'))
            if n%HZ==0:
                if team=='BALL':grids.setdefault(section,[]).append(n)
                elif person in keepers:gx.setdefault(section,[]).append((n,float(element.get('X'))))
            element.clear()
        elif event=='end' and element.tag=='FrameSet':element.clear()
    output={}
    for section,ns in grids.items():
        ns=np.array(sorted(ns));d=np.diff(ns)
        positions=sorted(gx.get(section,[]))
        output[section]=dict(samples=len(ns),first_frame=int(ns[0]),last_frame=int(ns[-1]),
            non_1s_steps=int((d!=HZ).sum()),max_step_seconds=float(d.max()/HZ),
            initial_keeper_side='right' if positions and positions[0][1]>0 else 'left' if positions else None,
            whole_period_keeper_side='right' if positions and np.mean([x for _,x in positions])>0 else 'left' if positions else None)
    return output


if __name__=='__main__':
    train,val,test=split(0);report={}
    for mid in train+[val,test]:
        report[mid]=audit(mid);print(mid,report[mid],flush=True)
    path=Path('cv/out/training_20261011/dfl_clock_audit.json');path.write_text(json.dumps(report,indent=2))
