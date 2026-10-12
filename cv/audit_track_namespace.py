"""Verify ID-only cache migration and inspect private export discontinuities."""
import html
import json
from pathlib import Path

import numpy as np

from cv.track_namespace import qualify_rows


def continuity(data):
    frames=data['frames']
    xy=np.array([f['xy'] for f in frames],float)
    obs=np.array([f['obs'] for f in frames],object)
    dt=np.diff([f['t'] for f in frames])[:,None]
    if not (dt>0).all():raise ValueError('Non-increasing export time')
    speeds=np.linalg.norm(np.diff(xy,axis=0),axis=2)/dt
    output={}
    for a,b,name in [(True,True,'measurement_measurement'),(True,False,'measurement_estimate'),
                      (False,True,'estimate_measurement'),(False,False,'estimate_estimate')]:
        selected=(obs[:-1]==a)&(obs[1:]==b)&np.isfinite(speeds)
        values=speeds[selected]
        output[name]=dict(n=len(values),over_10_m_s=int((values>10).sum()),mean_m_s=float(values.mean()) if len(values) else None,
            p90_m_s=float(np.percentile(values,90)) if len(values) else None,max_m_s=float(values.max()) if len(values) else None)
    return dict(observed_count=int((obs==True).sum()),continuity=output,proxy=data['meta']['source']['error_vs_cv_proxy']['reveals'])


def main():
    root=Path('cv/out/audit8_20261011/broadcast')
    results={};rows=[]
    for tracker in ['legacy','selected']:
        native=[json.loads(line) for line in (root/f'{tracker}_raw'/'detections.jsonl').open()]
        qualified=[json.loads(line) for line in (root/f'{tracker}_qualified'/'detections.jsonl').open()]
        assert qualify_rows(native)==qualified
        # Removing namespace fields must reproduce every original measurement.
        for original,copy in zip(native,qualified):
            assert {k:v for k,v in copy.items() if k!='shot_id' and k!='dets'}=={k:v for k,v in original.items() if k!='dets'}
            for a,b in zip(original['dets'],copy['dets']):
                assert {k:v for k,v in b.items() if k not in ('id','native_id','shot_id')}=={k:v for k,v in a.items() if k!='id'}
        for namespace in ['native','qualified']:
            name=f'{tracker}_{namespace}'
            results[name]=continuity(json.loads((root/f'{name}_match.json').read_text()))
            r=results[name];c=r['continuity']['measurement_measurement'];p=r['proxy']
            rows.append([name,r['observed_count'],c['over_10_m_s'],c['n'],round(c['max_m_s'],2),p['n'],p['mean_error_m']])
    result=dict(measurement_preservation='All cached fields except track IDs/provenance are exactly preserved',results=results,
        conditions='Same cached input within each tracker pair. Export role aliases and supported sets change. Continuity is not accuracy; references are unverified role-assigned CV detections, not ground truth. No active export is modified.')
    (root/'namespace_comparison.json').write_text(json.dumps(result,indent=2))
    headings=['Private candidate','Measured positions','Measured steps >10m/s','Measured transitions','Maximum m/s','Proxy samples','Proxy mean m']
    table='<table><thead><tr>'+''.join(f'<th>{h}</th>' for h in headings)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'
    (root/'namespace_comparison.html').write_text('<!doctype html><meta charset="utf-8"><title>Private identity namespace review</title><style>body{font:16px system-ui;margin:30px;background:#111827;color:#e2e8f0}table{border-collapse:collapse}th,td{border:1px solid #475569;padding:10px;text-align:left}</style><h1>Camera-cut identity namespaces</h1><p>'+html.escape(result['conditions'])+'</p><p>'+html.escape(result['measurement_preservation'])+'</p>'+table)
    print(rows,flush=True)


if __name__=='__main__':main()
