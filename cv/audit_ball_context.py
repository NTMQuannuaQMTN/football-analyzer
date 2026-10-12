"""Read-only private broadcast ball/context audit; no ground-truth accuracy claim.

Separates raw image candidates, stored pitch candidates, temporal acceptance,
and filled 1 Hz context. Reference labels never enter this diagnostic.
"""
import hashlib
import html
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from cv.build_match import ball_track
from cv.session_report import table

ROOT = Path('cv/out/audit8_20261011')


def context_sources(accepted, duration):
    """Mirror the exporter gap policy without calling fills measurements.

An accepted source somewhere in [s,s+1) is still a future-bucket average.
Short interpolation uses a future accepted bucket; other gaps hold a stale
coordinate (or the initial midfield value). All are retrospective context.
"""
    seconds = sorted({int(t) for t, *_ in accepted if 0 <= t < duration + 1})
    result = []
    for second in range(duration + 1):
        if second in seconds:
            result.append('accepted_detection_bucket')
            continue
        before = [s for s in seconds if s < second]
        after = [s for s in seconds if s > second]
        if before and after and after[0] - before[-1] <= 3:
            result.append('future_interpolation')
        elif before:
            result.append('held_previous_coordinate')
        else:
            result.append('initial_midfield_fallback')
    return result


def longest_missing_run(rows):
    """Count successive main-view samples without acceptance; cuts break runs."""
    longest = current = 0
    for main, accepted in rows:
        current = current + 1 if main and not accepted else 0
        longest = max(longest, current)
    return longest


def main():
    output = ROOT / 'ball_context'
    output.mkdir(parents=True, exist_ok=True)
    raw_root = ROOT / 'broadcast'
    source = raw_root / 'appearance_raw'
    frames = json.loads((raw_root / 'frames.json').read_text())
    rows = [json.loads(line) for line in (source / 'detections.jsonl').open()]
    if len(rows) != len(frames) or [r['t'] for r in rows] != [f['t'] for f in frames]:
        raise ValueError('Ball image and pitch candidates need identical timestamps')
    run = json.loads((source / 'run.json').read_text())
    candidates = [(r['t'] - run['start'], d['px'], d['py'], d['conf'])
                  for r in rows for d in r['dets'] if d['cls'] == 'ball']
    _, accepted = ball_track(candidates)
    accepted_times = {round(t + run['start'], 2) for t, *_ in accepted}
    sources = context_sources(accepted, int(run['end'] - run['start']))
    samples = []
    hashes = {str(source / 'detections.jsonl'): hashlib.sha256((source / 'detections.jsonl').read_bytes()).hexdigest(),
              str(raw_root / 'frames.json'): hashlib.sha256((raw_root / 'frames.json').read_bytes()).hexdigest(),
              'cv/build_match.py': hashlib.sha256(Path('cv/build_match.py').read_bytes()).hexdigest()}
    for frame, row in zip(frames, rows):
        raw = []
        if frame['image'] is not None:
            path = raw_root / 'raw' / (Path(frame['image']).stem + '.json')
            data = json.loads(path.read_text())
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            raw = data['balls']
        pitch = [d for d in row['dets'] if d['cls'] == 'ball']
        samples.append(dict(video_s=row['t'], main_camera=frame['main_camera'],
            camera_supported=row['live'], raw_candidates=len(raw),
            raw_score_ge_02=sum(d[4] >= .2 for d in raw),
            raw_max_score=max((d[4] for d in raw), default=None),
            pitch_candidates=len(pitch), temporal_accepted=round(row['t'], 2) in accepted_times,
            image=frame['image']))
    main_samples = [s for s in samples if s['main_camera']]
    summary = dict(all_timestamps=len(samples), main_camera_timestamps=len(main_samples),
        with_any_raw_ball_candidate=sum(s['raw_candidates'] > 0 for s in main_samples),
        with_raw_score_ge_02=sum(s['raw_score_ge_02'] > 0 for s in main_samples),
        with_multiple_raw_score_ge_02=sum(s['raw_score_ge_02'] > 1 for s in main_samples),
        with_pitch_ball_candidates=sum(s['pitch_candidates'] > 0 for s in main_samples),
        with_temporally_accepted_ball=sum(s['temporal_accepted'] for s in main_samples),
        accepted_detection_count=len(accepted), projected_candidate_count=len(candidates),
        longest_main_view_without_accepted_sample_count=longest_missing_run(
            [(s['main_camera'], s['temporal_accepted']) for s in samples]),
        sample_interval_s=1 / run['fps'], export_context_sources=dict(Counter(sources)))
    speeds = []
    for a, b in zip(accepted, accepted[1:]):
        dt = b[0] - a[0]
        if 0 < dt <= 1:
            speeds.append(dict(video_s=b[0]+run['start'], elapsed_s=dt,
                apparent_pitch_speed_m_s=float(np.linalg.norm(np.array(b[1:3])-a[1:3])/dt)))
    if speeds:
        values = [s['apparent_pitch_speed_m_s'] for s in speeds]
        summary['accepted_step_diagnostic'] = dict(n=len(values), mean_m_s=float(np.mean(values)),
            p90_m_s=float(np.percentile(values,90)), max_m_s=max(values),
            steps_over_30_m_s=sum(s > 30 for s in values))
    conditions = ('Same private cached stock YOLO combined predictions at 5 Hz; at most three ball '
        'candidates per image and score >=0.03. Candidate scores are not calibrated probabilities. '
        'Pitch candidates have already passed the unverified CCBV camera/field filter; ball_track '
        'adds the existing causal reachability rule, not a ball-identity guarantee. Main-view labels '
        'and camera source are retrospective. The 1 Hz exporter subsequently uses future bucket '
        'averaging/interpolation and inferred touches, nearest-player snapping, dead-ball and goal '
        'event constraints. These are estimates, not additional observed ball detections. '
        'No independent ball labels exist here. Ball in the air does not satisfy the pitch-plane '
        'projection assumption. Counts, apparent speeds and missing runs are diagnostics, not accuracy.')
    original = Path('data/cv_run3_yolo11m_backup.json')
    archive = json.loads(original.read_text())
    archive_summary = dict(frames=len(archive['frames']),
        with_finite_ball=sum(bool(f.get('b') is not None and np.isfinite(f['b']).all()) for f in archive['frames']),
        with_ball_source_label=sum('bs' in f for f in archive['frames']))
    hashes[str(original)] = hashlib.sha256(original.read_bytes()).hexdigest()
    result = dict(summary=summary, archived_export=archive_summary, conditions=conditions,
        samples=samples, accepted_step_diagnostics=speeds, sources_sha256=hashes,
        training_change='None', production_change='None')
    (output / 'results.json').write_text(json.dumps(result,indent=2))
    scalars = [[k,v] for k,v in summary.items() if not isinstance(v,dict)]
    page = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Private ball context audit</title>',
        '<style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:16px}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{border:1px solid #aaa;padding:8px}img{max-width:100%}</style>',
        '<h1>Private ball and possession context</h1><p>'+html.escape(conditions)+'</p>',
        table(['Availability diagnostic','Count / value'],scalars),
        '<h2>Before inferred-touch and event postprocessing</h2>',
        table(['1 Hz context source','Seconds'],sorted(Counter(sources).items())),
        '<h2>Original archived match</h2>', table(['Measure','Count'],archive_summary.items()),
        '<p>A finite ball on every replay frame does not establish an observed ball or reliable possession. '
        'Missingness needs its own inference feature and downstream evidence label before this context can '
        'support strong tactical statements. No export was changed.</p>']
    times = [1800.] + [r['video_s'] for r in sorted(speeds,key=lambda r:r['apparent_pitch_speed_m_s'],reverse=True)[:4]]
    seen = set()
    for t in times:
        frame = min((f for f in frames if f['image']),key=lambda f:abs(f['t']-t))
        if frame['image'] in seen:continue
        seen.add(frame['image']);image=cv2.imread(str(raw_root/'images'/frame['image']))
        data=json.loads((raw_root/'raw'/(Path(frame['image']).stem+'.json')).read_text())
        for i,b in enumerate(data['balls']):
            cv2.rectangle(image,(round(b[0]),round(b[1])),(round(b[2]),round(b[3])),(0,180,255),2)
            cv2.putText(image,f'candidate {i+1}: score {b[4]:.3f}',(max(5,round(b[0])-100),max(18,round(b[1])-8)),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,180,255),1)
        name=Path(frame['image']).stem+'.jpg';cv2.imwrite(str(output/name),image)
        page+=['<h2>Video second '+str(frame['t'])+'</h2><p>Orange marks unverified detector candidates, '
               'including low-score candidates; this is a diagnostic selection, not a ball-accuracy sample.</p><img src="'+name+'" alt="Private ball candidate boxes">']
    page.append('</html>');(output/'review.html').write_text(''.join(page))
    print(summary,archive_summary,output/'review.html',flush=True)


if __name__ == '__main__':
    main()
