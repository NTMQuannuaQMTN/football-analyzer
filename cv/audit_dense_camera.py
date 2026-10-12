"""Score cameras between image checkpoints, with feet used only after inference."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from cv.audit_camera_motion import stats
from cv.camera_motion import ground_motion,propagate
from cv.camera_quality import assess_camera,plausible
from cv.camera_similarity import similarity_fallback
from cv.cache_provenance import files_digest
from cv.gsr import GSR,load_clip,project
from cv.session_report import table


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--part',choices=['val','test'],required=True)
    parser.add_argument('--foreground-exclusion',action='store_true',help='Actual detector-box masks on the existing tracking-validation subset')
    parser.add_argument('--all-validation',action='store_true',help='Frozen-parameter validation expansion; fill missing stock CPU detection cache, no new selection')
    parser.add_argument('--visible-anchor-agreement',action='store_true',help='Private p90 camera-agreement check at current detected feet; separate output')
    parser.add_argument('--out',help='Separate private output directory for reproducibility checks')
    args=parser.parse_args();cv2.setNumThreads(1)
    if args.all_validation and (args.part!='val' or not args.foreground_exclusion):
        raise ValueError('Validation expansion requires foreground exclusion and the validation split')
    if args.visible_anchor_agreement and not args.foreground_exclusion:
        raise ValueError('Visible anchor agreement requires actual foreground detections')
    if args.visible_anchor_agreement and not args.out:
        raise ValueError('Visible anchor agreement requires --out to preserve the earlier foreground experiment')
    root=Path('cv/out/audit8_20261011');out=Path(args.out) if args.out else root/('dense_foreground_camera' if args.foreground_exclusion else 'dense_camera');out.mkdir(parents=True,exist_ok=True)
    if args.part=='test' and not json.loads((out/'gate_result.json').read_text())['passed']:
        raise ValueError('Dense validation gate did not pass; do not inspect held-out candidate')
    source=json.loads((root/'camera_geometric'/f'{args.part}.json').read_text())
    config=json.loads((root/'camera_motion/selection.json').read_text())['configuration']
    strict_hash=hashlib.sha256(Path('cv/camera_motion.py').read_bytes()).hexdigest()
    fallback_hash=hashlib.sha256((strict_hash+hashlib.sha256(Path('cv/camera_similarity.py').read_bytes()).hexdigest()).encode()).hexdigest()
    methods={'strict':(ground_motion,'camera_motion',strict_hash),'similarity_fallback':(similarity_fallback,'camera_similarity',fallback_hash)}
    candidate_name='similarity_fallback';allowed=None;raw_cache=None
    if args.foreground_exclusion:
        from cv.camera_foreground import motion_without_people
        tracker=json.loads((root/'tracking_generic'/f'{args.part}.json').read_text())
        allowed={r['clip'] for r in tracker['per_clip']}
        if args.all_validation:allowed=None
        raw_cache=root/'tracking_generic/cache'/f"{tracker['model_sha256'][:16]}_{tracker['device']}_{tracker['ultralytics']}_960_c015_i05"
        masked_hash=hashlib.sha256(Path('cv/camera_foreground.py').read_bytes()+Path('cv/camera_motion.py').read_bytes()).hexdigest()
        candidate_name='foreground_exclusion'
        methods={'strict':methods['strict'],candidate_name:(motion_without_people,'camera_foreground',masked_hash)}
        if args.visible_anchor_agreement:
            candidate_name='visible_anchor_agreement'
            methods[candidate_name]=(motion_without_people,'camera_foreground',masked_hash)
    indexed={(r['clip'],Path(r['image']).stem.split('_')[-1]):r for r in source['cases']}
    detector=None;filled=0
    errors={k:{g:[] for g in ['all','anchor_times','between_anchors','paired']} for k in methods}
    counts={k:0 for k in methods};good={k:0 for k in methods};total_people=0;cases=[];cache_inputs={};reference_sources={}
    for clip in sorted({r['clip'] for r in source['cases']}):
        if allowed is not None and clip not in allowed:continue
        _,frames=load_clip(GSR/'valid'/clip);previous=None;previous_boxes=None
        keys={name:method[2] for name,method in methods.items()}
        if args.foreground_exclusion:
            photos=[f['file'] for f in frames[::5]]
            raw_files=[raw_cache/f'{clip}_{f["file"].stem}.npy' for f in frames[::5]]
            if args.all_validation:
                for image_path,raw_path in zip(photos,raw_files):
                    if raw_path.exists():continue
                    if detector is None:
                        import torch
                        from ultralytics import YOLO
                        torch.set_num_threads(1)
                        detector=YOLO(tracker['model'])
                    people=[key for key,value in detector.names.items() if value in ('person','player','goalkeeper','referee')]
                    detections=detector.predict(str(image_path),classes=people,imgsz=960,conf=.15,iou=.5,device='cpu',verbose=False)[0]
                    np.save(raw_path,detections.boxes.data.cpu().numpy());filled+=1
            image_hash=files_digest(photos);detector_hash=files_digest(raw_files)
            cache_inputs[clip]=dict(images_sha256=image_hash,detections_sha256=detector_hash)
            keys={name:hashlib.sha256((method[2]+image_hash+(detector_hash if name!='strict' else '')).encode()).hexdigest() for name,method in methods.items()}
            reference_sources[clip]=hashlib.sha256((GSR/'valid'/clip/'Labels-GameState.json').read_bytes()).hexdigest()
        states={k:dict(h=None,anchor_time=-100.) for k in methods}
        for i,frame in enumerate(frames[::5]):
            time=i*.2;image=cv2.imread(str(frame['file']));row=indexed.get((clip,frame['file'].stem))
            raw=np.load(raw_cache/f'{clip}_{frame["file"].stem}.npy') if raw_cache is not None else None
            boxes=raw[:,:4] if raw is not None else None
            quality_image=cv2.imread(row['image']) if row else cv2.resize(image,(512,288))
            anchor=None
            if row:
                result=row['results']['learned_image_fit']
                if 'H' in result and plausible(assess_camera(quality_image,np.array(result['H'])),.75,.25):anchor=np.array(result['H'])
            predicted={}
            for name,(motion,cache_root,key) in methods.items():
                cache=root/cache_root/'flow'/keys[name][:16]/clip;cache.mkdir(parents=True,exist_ok=True)
                path=cache/f'{frame["file"].stem}.npz'
                if previous is None:warp=None
                elif path.is_file():
                    value=np.load(path);warp=value['h'] if bool(value['valid']) else None
                else:
                    warp,_=motion(previous,image,previous_boxes,boxes) if name!='strict' else motion(previous,image)
                    np.savez_compressed(path,h=warp if warp is not None else np.eye(3),valid=warp is not None)
                state=states[name];h=propagate(state['h'],warp)
                if time-state['anchor_time']>config['age']:h=None
                if anchor is not None:
                    accept=h is None
                    if not accept:
                        points=[[.2,.4],[.5,.4],[.8,.4],[.2,.8],[.5,.8],[.8,.8]]
                        delta=np.linalg.norm(project(h,points)-project(anchor,points),axis=1)
                        disagreement=float(np.median(delta)) if np.isfinite(delta).all() else float('inf')
                        if name=='visible_anchor_agreement':
                            from cv.visible_anchor_agreement import visible_foot_disagreement
                            measured=visible_foot_disagreement(h,anchor,raw,(image.shape[1],image.shape[0]))
                            if measured is not None:disagreement=measured
                        accept=disagreement<=config['agreement']
                    if accept:h=anchor.copy();state['anchor_time']=time
                state['h']=h
                predicted[name]=h if h is not None and plausible(assess_camera(quality_image,h),.75,.25) else None
            previous=image;previous_boxes=boxes
            # References enter only here, after both cameras have been fixed.
            reference=[p for p in frame['people'] if p['pitch'] is not None and p['role'] in ('player','goalkeeper','referee')]
            total_people+=len(reference);values={};case=dict(clip=clip,frame=frame['file'].stem,t=time,image_checkpoint=row is not None,people=len(reference),methods={})
            for name,h in predicted.items():
                error=None
                if h is not None and reference:
                    feet=[[(p['box'][0]+p['box'][2])/2/1920,p['box'][3]/1080] for p in reference]
                    candidate=np.linalg.norm(project(h,feet)-np.array([p['pitch'] for p in reference]),axis=1)
                    if np.isfinite(candidate).all():error=candidate
                values[name]=error
                if error is not None:
                    counts[name]+=1;good[name]+=int((error<=5).sum());errors[name]['all'].extend(error.tolist())
                    errors[name]['anchor_times' if row else 'between_anchors'].extend(error.tolist())
                case['methods'][name]=dict(supported=error is not None,camera_supported=h is not None,
                    H=h.tolist() if h is not None else None,errors_m=error.tolist() if error is not None else [])
            if all(v is not None for v in values.values()):
                for name,error in values.items():errors[name]['paired'].extend(error.tolist())
            cases.append(case)
        print(args.part,clip,len(cases),flush=True)
    summary={name:dict(total_frames=len(cases),supported_frames=counts[name],frame_coverage=counts[name]/len(cases),total_labelled_people=total_people,
        within_5m_recall_all_people=good[name]/total_people,errors={group:stats(values) for group,values in groups.items()}) for name,groups in errors.items()}
    conditions='Identical learned image-only checkpoint cameras and frozen age3s/agreement10m policy. Score every fifth 25Hz frame, including frames between 1.2s image checkpoints. Current/past flow only, independent visible reference feet read after camera inference. Exact labelled feet isolate cameras, not detector/named/hidden performance. The denominator includes all people with usable pitch labels, including camera failures; missing pitch labels cannot be scored. One held-out GSR game; not EPL calibration.'
    if args.foreground_exclusion:conditions+=' Actual cached generic detections mask people at both images; no labelled boxes enter flow. Validation uses the same eight tracking-validation clips; test uses all 18 held-out clips.'
    if args.all_validation:conditions+=' Additional frozen-parameter check on every available validation clip; supersedes the eight-clip scope for this file only. Missing CPU stock-detection caches filled with identical settings; no camera or tracker parameter selection.'
    if args.visible_anchor_agreement:conditions+=' Visible-anchor candidate uses p90 disagreement at at least four current confidence>=.3 detected feet, otherwise the unchanged grid check. Fixed 10m agreement and 3s age; no truth or future inputs. Missing flow still permits an anchor and remains a limitation.'
    (out/f'{args.part}.json').write_text(json.dumps(dict(part=args.part,configuration=config,summary=summary,cases=cases,conditions=conditions,
        camera_H_coordinates='normalised_0_to_1_image_to_pitch_metres',sampling_hz=5,
        inference_cache_inputs=cache_inputs,evaluation_reference_sha256=reference_sources,
        expanded_validation=args.all_validation,new_detection_cache_frames=filled,
        visible_anchor_agreement=args.visible_anchor_agreement,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    if args.part=='val':
        base=summary['foreground_exclusion' if args.visible_anchor_agreement else 'strict'];candidate=summary[candidate_name];b=base['errors']['all'];c=candidate['errors']['all']
        passed=bool(c['mean_m'] is not None and candidate['frame_coverage']>=base['frame_coverage'] and c['mean_m']<=b['mean_m']+.05 and c['p90_m']<=b['p90_m']+.1 and candidate['within_5m_recall_all_people']>=base['within_5m_recall_all_people']-.002)
        (out/'gate_result.json').write_text(json.dumps(dict(passed=passed,criterion='Frozen before dense scoring: coverage non-decreasing, mean no more than +.05m, p90 no more than +.1m, within-5m/all-person recall loss at most .2pp. No held-out parameter changes.',summary=summary),indent=2))
    rows=[]
    for name,r in summary.items():
        for group,e in r['errors'].items():rows.append([name,group,e['n'],round(e['mean_m'],3) if e['n'] else 'N/A',round(e['median_m'],3) if e['n'] else 'N/A',round(e['p90_m'],3) if e['n'] else 'N/A',f"{r['frame_coverage']:.2%}",f"{r['within_5m_recall_all_people']:.2%}"])
    (out/f'{args.part}.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Dense camera evaluation</title><style>body{font:16px system-ui;margin:30px}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:8px}</style><h1>Dense camera evaluation: '+args.part+'</h1><p>'+conditions+'</p>'+table(['Camera motion','Scored group','Feet','Mean m','Median m','P90 m','All-frame coverage','Within-5m / all people'],rows)+'</html>')
    print(summary,flush=True)


if __name__=='__main__':main()
