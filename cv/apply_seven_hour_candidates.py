"""Apply frozen experimental checkpoints beside originals; score DFL against truth."""
import argparse
import json
import shutil
import subprocess
import sys
import time
import os
from pathlib import Path

import numpy as np

from backend.reconstruction import dfl_match
from backend.reconstruction.datasets import dfl_matches
from cv.session_report import table


def coordinates(frames):
    return np.asarray([[p if p is not None else [np.nan,np.nan] for p in frame['xy']] for frame in frames],float)


def position_metrics(data,mode,require_preserved_observations=True):
    truth=coordinates(data['frames'])
    prediction=coordinates(data['alt_frames'][mode])
    observed=np.asarray([f['obs'] for f in data['alt_frames'][mode]],bool)
    valid=np.isfinite(truth).all(2)&np.isfinite(prediction).all(2)
    error=np.linalg.norm(prediction-truth,axis=2)
    out={}
    for name,mask in [('all',valid),('observed',valid&observed),('hidden',valid&~observed)]:
        values=error[mask];out[name]=dict(n=len(values),mean_m=float(values.mean()) if len(values) else None,median_m=float(np.median(values)) if len(values) else None,p90_m=float(np.percentile(values,90)) if len(values) else None)
    out['observed_coordinates_preserved']=not bool((error[valid&observed]>1e-3).any())
    if require_preserved_observations:
        assert out['observed_coordinates_preserved'],'Visible ground-truth measurements were mutated'
    last=np.full(truth.shape[1:],np.nan);lop=[];paired=[];ages=np.full(truth.shape[1],np.inf);groups={}
    for t in range(len(truth)):
        seen=observed[t]&valid[t];last[seen]=truth[t,seen];ages[seen]=0
        hidden=valid[t]&~observed[t];usable=hidden&np.isfinite(last).all(1)
        if usable.any():lop.extend(np.linalg.norm(last[usable]-truth[t,usable],axis=1).tolist());paired.extend(error[t,usable].tolist())
        for label,mask in [('0–3 s',ages<=3),('3–10 s',(ages>3)&(ages<=10)),('10–30 s',(ages>10)&(ages<=30)),('>30 s',np.isfinite(ages)&(ages>30)),('Never observed',~np.isfinite(ages))]:
            groups.setdefault(label,[]).extend(error[t,hidden&mask].tolist())
        ages+=1
    out['by_observation_age']={k:dict(n=len(v),mean_m=float(np.mean(v)) if v else None,p90_m=float(np.percentile(v,90)) if v else None) for k,v in groups.items()}
    out['last_observed_paired']=dict(n=len(lop),last_observed_mean_m=float(np.mean(lop)) if lop else None,model_mean_m=float(np.mean(paired)) if paired else None)
    radius=np.asarray([f.get('rad',[None]*truth.shape[1]) for f in data['alt_frames'][mode]],float)
    covered=valid&~observed&np.isfinite(radius)
    out['uncertainty_coverage']=dict(n=int(covered.sum()),inside_radius_fraction=float((error[covered]<=radius[covered]).mean()) if covered.any() else None,basis='Conditional on exact visible tracking, identities/roles and declared context; not broadcast perception uncertainty')
    return out


def atomic_export(data,path):
    path.parent.mkdir(exist_ok=True,parents=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,separators=(',',':')));tmp.replace(path)


def export_dfl(root,training,phase):
    records=[];dfl_match.PIN_MODE='off'
    for i,mid in enumerate(sorted(dfl_matches())):
        fold=training/f'fold_{i}';status=json.loads((fold/'training_results.json').read_text())
        variant='robust' if all(status['mode_validation_gates'].values()) else 'clean_control'
        tag=f'robustness_{variant}_v3'
        for half in (1,2):
            output=dfl_match.build(mid,half=half,seed=0,fold=tag,weights_dir=fold)
            output['meta']['source'].update(experimental=True,session='seven-hour',phase=phase,checkpoint_directory=str(fold),validation_gates=status['mode_validation_gates'],event_pins_disabled=True)
            name=f'dfl_{mid.split("-")[-1]}_trial7h_{phase}_h{half}.json';path=Path('data')/name
            reference=Path('data')/f'dfl_{mid.split("-")[-1]}_h{half}.json'
            baseline=None;alignment='Original unavailable'
            if reference.exists():
                original=json.loads(reference.read_text())
                if original['meta']['player_order']==output['meta']['player_order'] and len(original['frames'])==len(output['frames']):
                    same=np.allclose(coordinates(original['frames']),coordinates(output['frames']),equal_nan=True,atol=.001)
                    if same:baseline={mode:position_metrics(original,mode,require_preserved_observations=False) for mode in ('live','replay')};alignment='Same truth/order/time; legacy event postprocessing may differ, including modified observed coordinates (scored separately)'
                    else:alignment='Ground-truth coordinate alignment differs; comparison withheld'
                else:alignment='Roster/time alignment differs; comparison withheld'
            metrics={mode:position_metrics(output,mode) for mode in ('live','replay')}
            atomic_export(output,path);records.append(dict(match=mid,half=half,file=str(path),phase=phase,variant=variant,baseline=baseline,metrics=metrics,comparison_conditions=alignment))
            (root/f'dfl_{phase}_metrics.json').write_text(json.dumps(records,indent=2));print('applied',path,metrics['live']['hidden'],flush=True)
    return records


def export_epl(root,training,phase):
    folder=root/f'epl_{phase}';folder.mkdir(exist_ok=True)
    checkpoint=Path('cv/out/closeup_retrain_20261012_015904/camera/best.pt')
    # Freeze complete files only. An in-progress best.pt is not a candidate snapshot.
    new=root/'camera_markings/best.pt'
    if (root/'camera_markings/finished.json').exists() and new.exists():checkpoint=new
    frozen=folder/'camera.pt';shutil.copy2(checkpoint,frozen)
    command=[sys.executable,'-u','-m','cv.reproject_camera_review','--out',str(folder/'measurements'),'--exclude-detected-people','--learned-checkpoint',str(frozen),'--view-checkpoint','cv/out/closeup_retrain_20261012_015904/views/best.pt']
    subprocess.run(command,check=True)
    fold=training/'fold_0';status=json.loads((fold/'training_results.json').read_text());variant='robust' if all(status['mode_validation_gates'].values()) else 'clean_control'
    from cv.build_match import build
    destination=folder/'match.json';build(folder/'measurements','home',destination,f'robustness_{variant}_v3',fold,'nearest','stable_track','shot')
    data=json.loads(destination.read_text());data['meta']['source'].update(experimental=True,phase=phase,session='seven-hour',provider='SoccerNet · experimental',camera_checkpoint=str(frozen),view_classifier='Frozen same-match diagnostic classifier; probabilities uncalibrated')
    published=Path('data')/f'cv_trial7h_arsenal_{phase}.json';atomic_export(data,published)
    from cv.private_review import render
    for label,start in [('start',0),('penalty',298),('goal',365)]:
        try:render(published,folder/f'{label}_private.mp4',start,10,fps=10)
        except Exception as error:(folder/f'{label}_movie_error.txt').write_text(str(error))
    print('EPL candidate published',published,flush=True);return str(published)


def exported_epl_candidates(root,training,phase):
    from cv.clip_eval import load
    from backend.reconstruction.inference import reconstruct_ml
    from backend.reconstruction.postprocess import limit_speed
    fold=training/'fold_0';status=json.loads((fold/'training_results.json').read_text());variant='robust' if all(status['mode_validation_gates'].values()) else 'clean_control'
    files=[]
    for path in sorted(Path('data').glob('cv_skillcorner_epl_*_h*.json')):
        data,m,vis,on=load(path);teams=np.array([o[0] for o in m.order]);coarse=candidate_roles(data['players']);rc=np.array([coarse[o] for o in m.order]);roles={o:'GK' if coarse[o]=='GK' else 'CM' for o in m.order}
        m.xy=np.where(vis[...,None],m.xy,np.nan)
        rec=reconstruct_ml(m,vis,'replay',on,roles,None,teams,rc,fold=f'robustness_{variant}_v3',weights_dir=fold)
        pos=limit_speed(rec.pos,vis,on,causal=False)
        for t,frame in enumerate(data['frames']):
            for j in range(len(m.order)):
                if not vis[t,j] and on[t,j] and np.isfinite(pos[t,j]).all():frame['xy'][j]=np.round(pos[t,j],1).tolist()
            frame['rad']=[0. if vis[t,j] else float(rec.rad[t,j]) if on[t,j] else None for j in range(len(m.order))]
        source=data['meta']['source'];source.update(experimental=True,phase=phase,session='seven-hour',reconstruction_weights=str(fold),reconstruction_fold=f'robustness_{variant}_v3',uncertainty_basis='DFL calibration transferred; video-domain uncertainty unverified')
        source.pop('error_vs_truth',None);source.pop('error_vs_cv_proxy',None)
        destination=Path('data')/f'cv_trial7h_{path.stem.removeprefix("cv_skillcorner_epl_")}_{phase}.json';atomic_export(data,destination);files.append(str(destination))
    (root/f'other_epl_{phase}.json').write_text(json.dumps({'applied_files':files,'conditions':'Stored measured positions only are inference inputs; hidden export estimates masked. No corresponding local video/full hidden truth, so player accuracy not scored.'},indent=2))


def candidate_roles(players):
    from cv.clip_eval import coarse_roles
    if all(p['role_slot']=='GK' or p['role_slot'].startswith('L') and p['role_slot'][1:].isdigit() for p in players):
        return coarse_roles(players)
    mapping={'GK':'GK','CF':'ST','ST':'ST','LW':'W','RW':'W','LM':'WM','RM':'WM',
        'LWB':'FB','RWB':'FB','LB':'FB','RB':'FB','LCB':'CB','RCB':'CB','CB':'CB',
        'CM':'CM','DM':'DM','AM':'AM'}
    return {p['id']:mapping[p['role_slot']] for p in players}


def refresh_demo(root,phase,epl):
    # run.sh independently checks ownership of both listeners before TERM.
    env=dict(os.environ,MATCH_FILE=epl,NARRATIVE_BACKEND='local')
    log=(root/f'demo_{phase}.log').open('w')
    child=subprocess.Popen(['bash','run.sh'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (root/f'demo_{phase}_process.json').write_text(json.dumps({'pid':child.pid,'start_epoch':time.time(),'match_file':epl,'note':'Owned demo process independent of training supervisor; guard refuses unrelated listeners'}))


def review(root,records,epl,phase):
    rows=[]
    for r in records:
        for mode,m in r['metrics'].items():
            values=m['hidden'];b=r['baseline'][mode]['hidden'] if r['baseline'] else None
            rows.append([r['match'],r['half'],mode,values['n'],round(values['mean_m'],3),round(values['median_m'],3),round(values['p90_m'],3),round(b['mean_m'],3) if b else 'N/A',r['variant']])
    html='<html lang="en"><meta charset="utf-8"><title>Seven-hour match application</title><style>body{font:16px system-ui;max-width:1300px;margin:28px}td,th{border:1px solid #aaa;padding:8px}table{border-collapse:collapse}</style><h1>Experimental match reconstruction</h1><p>DFL: exact visible tracking and simulated camera masks; held-out match checkpoints. Live and retrospective results separate. Legacy postprocessing differs; identical truth/time/player alignment checked. EPL: no full hidden truth, named identity/calibration still unverified. Broadcast movies remain private.</p>'+table(['DFL match','Half','Mode','Hidden N','Mean m','Median m','P90 m','Legacy mean m','Variant'],rows)+f'<p>EPL candidate: {epl}. Phase: {phase}. Original exports preserved. Reload the app/catalog after publishing.</p></html>'
    (root/f'review_{phase}.html').write_text(html)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    prior=Path('cv/out/audit8_20261011/context_crossvalidation')
    cache=a.root/'dfl_preview_metrics.json';records=json.loads(cache.read_text()) if cache.exists() else []
    if len(records)!=14 or not all(Path(r['file']).exists() for r in records):records=export_dfl(a.root,prior,'preview')
    epl='data/cv_trial7h_arsenal_preview.json'
    frozen=str(a.root/'epl_preview/camera.pt')
    existing=json.loads(Path(epl).read_text()) if Path(epl).exists() else None
    if existing is None or existing['meta']['source'].get('camera_checkpoint')!=frozen:epl=export_epl(a.root,prior,'preview')
    exported_epl_candidates(a.root,prior,'preview');review(a.root,records,epl,'preview');refresh_demo(a.root,'preview',epl)
    (a.root/'preview_applied.json').write_text(json.dumps({'DFL_parts':len(records),'EPL':epl,'finished_epoch':time.time()}))
    deadline=json.loads((a.root/'session.json').read_text())['deadline_epoch'];training=a.root/'dfl_training'
    while time.time()<deadline-300:
        if (training/'finished.json').exists():
            status=json.loads((training/'finished.json').read_text())
            if status['complete_training_folds']==7:break
            (a.root/'final_application_blocker.json').write_text(json.dumps(status));return
        time.sleep(30)
    if not (training/'finished.json').exists():return
    records=export_dfl(a.root,training,'trained');epl=export_epl(a.root,training,'trained');exported_epl_candidates(a.root,training,'trained');review(a.root,records,epl,'trained');refresh_demo(a.root,'trained',epl)
    (a.root/'trained_applied.json').write_text(json.dumps({'DFL_parts':len(records),'EPL':epl,'finished_epoch':time.time()}))


if __name__=='__main__':main()
