"""Inventory cached detections separately from confirmed pitch measurements.

Counts are not player recall: no independent visible or hidden EPL labels exist.
This module only reads private caches and writes private review artifacts.
"""
import hashlib
import html
import json
import os
from pathlib import Path

ROOT = Path('cv/out/audit8_20261011')
os.environ.setdefault('MPLCONFIGDIR', str((ROOT / 'mplconfig').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from cv.session_report import table


def main():
    output = ROOT / 'broadcast_coverage'
    output.mkdir(parents=True, exist_ok=True)
    sources = {}
    rows = []
    results = {}
    for name in ('legacy_qualified', 'selected_qualified', 'appearance_raw'):
        path = ROOT / 'broadcast' / name / 'detections.jsonl'
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        frames = [json.loads(line) for line in path.open()]
        cases = []
        for frame in frames:
            confirmed = [d for d in frame['dets'] if d['cls'] == 'person']
            unmatched = [d for d in frame.get('unassigned_dets', []) if d.get('cls') == 'person']
            cases.append(dict(video_s=frame['t'], main_camera=frame['live'],
                projected_tracks=len(confirmed), unassigned_boxes=len(unmatched),
                unassigned_missing_conf=sum('conf' not in d for d in unmatched),
                unassigned_conf_ge_05=sum(d['conf'] >= .5 for d in unmatched if 'conf' in d)))
        supported = [c for c in cases if c['main_camera']]
        counts = np.array([c['projected_tracks'] for c in supported])
        totals = dict(timestamps=len(cases), main_camera_timestamps=len(supported),
            projected_person_measurements=int(counts.sum()),
            median_projected_people=float(np.median(counts)),
            zero_confirmed_main_frames=int((counts == 0).sum()),
            unassigned_person_boxes=sum(c['unassigned_boxes'] for c in supported),
            unassigned_high_confidence_boxes=sum(c['unassigned_conf_ge_05'] for c in supported),
            unassigned_missing_conf=sum(c['unassigned_missing_conf'] for c in supported))
        results[name] = dict(summary=totals, cases=cases)
        rows.append([name, totals['main_camera_timestamps'], totals['projected_person_measurements'],
            totals['median_projected_people'], totals['zero_confirmed_main_frames'],
            totals['unassigned_person_boxes'], totals['unassigned_high_confidence_boxes'], totals['unassigned_missing_conf']])
    # Verify every branch uses the same camera flag and timestamps before comparing.
    reference = results['legacy_qualified']['cases']
    for name, result in results.items():
        assert [(c['video_s'], c['main_camera']) for c in result['cases']] == [
            (c['video_s'], c['main_camera']) for c in reference], name
    focus = []
    for name, result in results.items():
        for c in result['cases']:
            if abs(c['video_s'] - 1800) < .001:
                focus.append([name, c['video_s'], c['projected_tracks'], c['unassigned_boxes'], c['unassigned_conf_ge_05']])
    description = ('Identical cached stock YOLO detections and camera flags; tracker branches differ. '
        'Projected counts include anonymous people, potentially officials and false positives, '
        'not verified named players. Unassigned boxes may be unconfirmed tracks or rejected '
        'camera/field projections. Rejected projection records without scores are counted separately, '
        'never assigned an invented score. Missing observations do not prove a player is outside the '
        'camera. Confidence is a detector score, not calibrated position accuracy. No reference '
        'labels or future frames enter these counts; no full-player recall is claimed.')
    (output / 'results.json').write_text(json.dumps(dict(results=results, sources_sha256=sources,
        conditions=description), indent=2))
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for name, result in results.items():
        cases = result['cases']; x = np.array([c['video_s'] - 1440 for c in cases])
        for ax, key in zip(axes, ('projected_tracks', 'unassigned_conf_ge_05')):
            y = np.array([c[key] if c['main_camera'] else np.nan for c in cases])
            ax.plot(x, y, label=name, linewidth=.7, alpha=.8)
    axes[0].set_ylabel('Confirmed projected people')
    axes[1].set_ylabel('Unassigned boxes, score ≥0.5')
    axes[1].set_xlabel('Clip time (s); gaps denote unsupported camera views')
    axes[0].legend(fontsize=8)
    fig.suptitle('Private Arsenal coverage diagnostic: missing observations are not off-camera truth')
    fig.tight_layout(); fig.savefig(output / 'timeline.png', dpi=160); plt.close(fig)
    page = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Arsenal observation coverage</title>'
        '<style>body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:16px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #aaa;padding:8px}img{max-width:100%}</style>'
        '<h1>Private Arsenal observation coverage</h1><p>' + html.escape(description) + '</p>' +
        table(['Tracker','Main-camera frames','Projected observations','Median people/frame',
               'Main frames with zero tracks','Unassigned boxes','Unassigned score ≥0.5','Unassigned score N/A'], rows) +
        '<h2>At video second 1800</h2>' + table(['Tracker','Video s','Projected people','Unassigned boxes','Unassigned score ≥0.5'], focus) +
        '<img src="timeline.png" alt="Confirmed people and unassigned high-scoring boxes over time">'
        '<p>Watch <a href="../broadcast/anonymous_review.mp4">anonymous measurements</a> separately '
        'from <a href="../broadcast/robust_role_review.mp4">unverified role-assigned estimates</a>. '
        'Keep these local; neither is an accuracy-certified EPL reconstruction.</p></html>')
    (output / 'review.html').write_text(page)
    print(rows, focus, output / 'review.html', flush=True)


if __name__ == '__main__':
    main()
