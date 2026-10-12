"""Additive retrospective bridge and measured-anchor conflict diagnostics."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from backend.reconstruction.replay_bridge import bridge
from cv.apply_seven_hour_candidates import coordinates,atomic_export
from cv.session_report import table

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args()
    if a.input.resolve()==a.output.resolve():raise ValueError('Preserve original export; choose an additive output')
    d=json.loads(a.input.read_text());frames=d['frames'];p=coordinates(frames);v=np.array([[o is True for o in f['obs']] for f in frames]);new,report=bridge(p,v)
    assert np.array_equal(p[v],new[v])
    for t,f in enumerate(frames):
        for j in range(len(f['xy'])):
            if not v[t,j] and np.isfinite(new[t,j]).all():f['xy'][j]=np.round(new[t,j],1).tolist()
    for c in report['conflicts']:c['player_slot']=d['meta']['player_order'][c['player_index']]
    report.update(input=str(a.input),input_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),accuracy='No hidden EPL ground truth; continuity only, not position accuracy')
    mask=~(v[1:]&v[:-1]);before=np.linalg.norm(np.diff(p,axis=0),axis=2);after=np.linalg.norm(np.diff(new,axis=0),axis=2)
    report['hidden_related_steps_over_10m_s_before']=int(((before>10)&mask).sum());report['hidden_related_steps_over_10m_s_after']=int(((after>10)&mask).sum())
    src=d['meta']['source'];src.update(experimental=True,replay_bridge=report,phase='replay_bridge');src.pop('error_vs_truth',None);src.pop('error_vs_cv_proxy',None)
    atomic_export(d,a.output);a.report.mkdir(parents=True,exist_ok=True);(a.report/'bridge.json').write_text(json.dumps(report,indent=2))
    rows=[[c['player_slot'],c['last_t'],c['next_t'],round(c['required_mean_speed_m_s'],1),c['reason']] for c in report['conflicts']]
    (a.report/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Replay continuity conflicts</title><style>body{font:16px system-ui;margin:25px}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:8px}</style><h1>Measured-anchor conflicts</h1><p>Named slots remain unverified. Original measurements preserved. Feasible paths use later sightings; incompatible observations require camera/identity/time review. '+report['accuracy']+'</p>'+table(['Slot','Last clip s','Next clip s','Required m/s','Reason'],rows))
    print('feasible gaps',report['feasible_gaps'],'conflicts',len(report['conflicts']),'jumps',report['hidden_related_steps_over_10m_s_before'],report['hidden_related_steps_over_10m_s_after'],flush=True)
if __name__=='__main__':main()
