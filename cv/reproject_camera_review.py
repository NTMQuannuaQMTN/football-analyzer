"""Private camera-only ablation on frozen anonymous Arsenal measurements.

Uses the unchanged GSR validation-selected age/agreement policy. No detector,
tracker, role or hidden-position model is rerun; no active export is written.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from cv.camera_continuity import CameraContinuity
from cv.camera_motion import ground_motion
from cv.camera_quality import assess_camera,plausible
from cv.cache_provenance import files_digest
from cv.soccernet import Calibration,DEFAULT_GAME


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default='cv/out/audit8_20261011/camera_continuity/appearance_raw')
    parser.add_argument('--anchor-interval-s',type=float,default=0.,help='0: historical nearest-time anchors; 1: anchors only at integer video times, exact 2Hz sample/grid intersections')
    parser.add_argument('--similarity-fallback',action='store_true',help='Private bounded-motion experiment; use a separate output directory')
    parser.add_argument('--exclude-detected-people',action='store_true',help='Private foreground-masked ground-flow experiment; separate output required')
    parser.add_argument('--learned-checkpoint',help='Experimental image-only line camera; never uses reference landmarks')
    parser.add_argument('--view-checkpoint',help='Frozen view-category head; unsupported categories do not receive a sideline camera')
    args=parser.parse_args()
    if not np.isfinite(args.anchor_interval_s) or args.anchor_interval_s<0:raise ValueError('Invalid anchor interval')
    root=Path('cv/out/audit8_20261011');source=root/'broadcast/appearance_raw'
    output=Path(args.out)
    motion=ground_motion
    if args.similarity_fallback and args.exclude_detected_people:raise ValueError('Evaluate motion experiments separately')
    if args.similarity_fallback:
        from cv.camera_similarity import similarity_fallback
        motion=similarity_fallback
        if output.resolve()==(root/'camera_continuity/appearance_raw').resolve():raise ValueError('Preserve the original temporal-camera review')
    if args.exclude_detected_people:
        from cv.camera_foreground import motion_without_people
        motion=motion_without_people
        if output.resolve()==(root/'camera_continuity/appearance_raw').resolve():raise ValueError('Preserve the original temporal-camera review')
    if output.resolve()==source.resolve():raise ValueError('Camera review must not overwrite its source detections')
    output.mkdir(parents=True,exist_ok=True)
    config=json.loads((root/'camera_motion/selection.json').read_text())['configuration']
    policy=CameraContinuity(config['age'],config['agreement']);cv2.setNumThreads(1)
    camera=Calibration(DEFAULT_GAME,1)
    learned=None;view_encoder=None;view_head=None
    if args.learned_checkpoint:
        import torch
        from cv.train_camera_audit import ResNetPitch
        from cv.pitch_lines import heatmap_points,homography_from_lines
        torch.set_num_threads(1);learned=ResNetPitch()
        learned.load_state_dict(torch.load(args.learned_checkpoint,weights_only=False)['state']);learned.eval()
    if args.view_checkpoint:
        import torch
        from torchvision.models import resnet18
        view_state=torch.load(args.view_checkpoint,weights_only=False)
        view_encoder=resnet18(weights=None);view_encoder.load_state_dict(torch.load(view_state['encoder_path'],weights_only=True));view_encoder.fc=torch.nn.Identity();view_encoder.eval()
        view_head=torch.nn.Linear(512,1);view_head.load_state_dict(view_state['state']);view_head.eval()
    frames=json.loads((root/'broadcast/frames.json').read_text())
    rows=[json.loads(line) for line in (source/'detections.jsonl').open()]
    assert len(rows)==len(frames)
    previous=None;previous_boxes=None;diagnostics=[];measurements=0;input_hashes=None
    flow_hash=hashlib.sha256(Path('cv/camera_motion.py').read_bytes()).hexdigest()
    if args.similarity_fallback:flow_hash=hashlib.sha256((flow_hash+hashlib.sha256(Path('cv/camera_similarity.py').read_bytes()).hexdigest()).encode()).hexdigest()
    if args.exclude_detected_people:
        images=[root/'broadcast/images'/f['image'] for f in frames if f['image'] is not None]
        raw_files=[root/'broadcast/raw'/(Path(f['image']).stem+'.json') for f in frames if f['image'] is not None]
        input_hashes=dict(images_sha256=files_digest(images),detections_sha256=files_digest(raw_files))
        flow_hash=hashlib.sha256((flow_hash+hashlib.sha256(Path('cv/camera_foreground.py').read_bytes()).hexdigest()+json.dumps(input_hashes,sort_keys=True)).encode()).hexdigest()
    cache=root/'camera_continuity/flow'/flow_hash[:16];cache.mkdir(parents=True,exist_ok=True)
    with (output/'detections.jsonl').open('w') as stream:
        for frame,row in zip(frames,rows):
            assert abs(frame['t']-row['t'])<.001
            image=cv2.imread(str(root/'broadcast/images'/frame['image'])) if frame['image'] is not None else None
            boxes=None
            if image is not None and view_encoder is not None:
                canvas=np.zeros((224,224,3),np.uint8);canvas[49:175]=cv2.cvtColor(cv2.resize(image,(224,126)),cv2.COLOR_BGR2RGB)
                tensor=torch.from_numpy(canvas.transpose(2,0,1).copy()).float()[None]/255
                with torch.no_grad():
                    feature=view_encoder((tensor-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None])
                    probability=float(view_head((feature-torch.from_numpy(view_state['center']))/torch.from_numpy(view_state['scale'])).sigmoid()[0,0])
                if probability<(view_state['threshold'] if view_state['threshold'] is not None else .5):image=None
            if args.exclude_detected_people and image is not None:
                raw=json.loads((root/'broadcast/raw'/(Path(frame['image']).stem+'.json')).read_text())
                boxes=np.asarray(raw['people'],float).reshape(-1,6)[:,:4]
            warp=None;flow=dict(status='unsupported',reason='no previous main image')
            if image is not None and previous is not None:
                path=cache/(Path(frame['image']).stem+'.npz')
                if path.is_file():
                    v=np.load(path,allow_pickle=True);warp=v['h'] if bool(v['valid']) else None;flow=v['diagnostic'].item()
                else:
                    warp,flow=motion(previous,image,previous_boxes,boxes) if args.exclude_detected_people else motion(previous,image)
                    np.savez_compressed(path,h=warp if warp is not None else np.eye(3),valid=warp is not None,diagnostic=np.array(flow,dtype=object))
            previous=image;previous_boxes=boxes
            anchor_time=args.anchor_interval_s==0 or abs(row['t']/args.anchor_interval_s-round(row['t']/args.anchor_interval_s))<1e-6
            if learned is not None and image is not None and anchor_time:
                tensor=torch.from_numpy(cv2.cvtColor(cv2.resize(image,(512,288)),cv2.COLOR_BGR2RGB).transpose(2,0,1).copy()).float()[None]/255
                with torch.no_grad():probability=learned(tensor).sigmoid()[0].numpy()
                anchor=homography_from_lines(heatmap_points(probability))
            else:anchor=camera.at(row['t']) if anchor_time and learned is None else None
            if anchor is not None and image is not None:
                if learned is None:anchor=anchor@np.diag([1280.,720.,1.])
                if not plausible(assess_camera(image,anchor),.75,.25):anchor=None
            else:anchor=None
            h,evidence=policy.step(row['t'],anchor,warp,supported_view=image is not None)
            if h is not None and not plausible(assess_camera(image,h),.75,.25):
                h=None;policy.h=None;evidence.update(status='unsupported',reason='propagated camera failed frozen physical check')
            detections=[];unassigned=list(row.get('unassigned_dets',[]))
            for d in row['dets']:
                copy=dict(d);box=d.get('raw_box',d['box'])
                foot=[(box[0]+box[2])/2,box[3] if d['cls']!='ball' else (box[1]+box[3])/2]
                point=Calibration.project(h@np.diag([1/1280.,1/720.,1.]),[foot])[0] if h is not None else [np.nan,np.nan]
                if not np.isfinite(point).all() or not (-3<point[0]<108 and -3<point[1]<71):
                    unassigned.append(dict(id=None,track_id=d['id'],cls=d['cls'],box=box,conf=d['conf'],reason='camera-only candidate projection unsupported'))
                    continue
                copy.update(px=round(float(point[0]),2),py=round(float(point[1]),2),camera_evidence=evidence['status'])
                detections.append(copy);measurements+=d['cls']=='person'
            result={**row,'live':h is not None,'dets':detections,'unassigned_dets':unassigned,
                    'camera_evidence':evidence,'H':h.tolist() if h is not None else None}
            stream.write(json.dumps(result,allow_nan=False)+'\n')
            diagnostics.append(dict(t=row['t'],**evidence,flow=flow))
            if len(diagnostics)%200==0:print('camera',len(diagnostics),len(rows),measurements,flush=True)
    metadata=json.loads((source/'run.json').read_text())
    metadata.update(camera='Experimental guarded CCBV + current/past ground propagation',
        learned_checkpoint=args.learned_checkpoint,view_checkpoint=args.view_checkpoint,
        camera_configuration=config,source_detections_sha256=hashlib.sha256((source/'detections.jsonl').read_bytes()).hexdigest(),
        anchor_interval_s=args.anchor_interval_s,
        similarity_fallback=args.similarity_fallback,
        foreground_exclusion=args.exclude_detected_people,
        motion_input_sha256=input_hashes,
        motion_implementation_sha256=hashlib.sha256(Path('cv/camera_foreground.py').read_bytes()).hexdigest() if args.exclude_detected_people else None,
        camera_H_coordinates='normalised_0_to_1_image_to_pitch_metres',
        reprojection_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        camera_source_sha256={name:hashlib.sha256(Path('cv',name).read_bytes()).hexdigest() for name in ['camera_continuity.py','camera_motion.py','soccernet.py']},
        camera_conditions='External CCBV nearest samples and shot labels are retrospective. Image propagation reads past/current only. Same stored anonymous detections; no hidden truth, named identity or production promotion. Camera failures are explicit, not moved outside a polygon.',
        camera_support_gaps='Preserved qualified tracker IDs; new projection abstentions do not imply actual tracker resets')
    (output/'run.json').write_text(json.dumps(metadata,indent=2))
    if learned is not None:
        metadata.update(camera='Experimental image-only learned lines plus current/past ground propagation',calibration_sampling='Current cached image and past propagation; no CCBV camera substitution',camera_conditions='Frozen image-only checkpoint; no reference coordinates or future images. Shot annotations and export remain retrospective. Category scores do not establish calibration or identity accuracy.')
        (output/'run.json').write_text(json.dumps(metadata,indent=2))
    (output.parent/'reprojection_source.py').write_bytes(Path(__file__).read_bytes())
    (output.parent/'camera_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    print('Finished',output,'person measurements',measurements,flush=True)


if __name__=='__main__':main()
