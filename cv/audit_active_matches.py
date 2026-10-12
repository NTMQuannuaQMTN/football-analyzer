"""Read-only contract and downstream smoke audit of every active file export."""
import gc
import hashlib
from html import escape
import json
import os
from pathlib import Path
import time

import numpy as np

# This audit never invokes an optional paid narrator.
os.environ['NARRATIVE_BACKEND'] = 'local'
from backend.catalog import video_details
from backend.pipeline import MatchIntelligence


def audit(path):
    started = time.time()
    raw = json.loads(path.read_text())
    meta = raw['meta']
    order = meta['player_order']
    issues = []
    times = np.asarray([f['t'] for f in raw['frames']], float)
    if len(set(order)) != len(order):
        issues.append('Duplicate player IDs')
    if len(times) != int(meta['duration'])+1 or not np.array_equal(times, np.arange(len(times))):
        issues.append('Timeline differs from the engine integer-second indexing contract')
    totals = dict(present=0, measured=0, estimated=0, nonfinite=0, outside_pitch=0, measured_jump_over_10m_s=0, measured_transitions=0)
    previous = None
    for frame in raw['frames']:
        for field in ['xy', 'obs', 'conf', 'rad', 'observed_t']:
            if field in frame and len(frame[field]) != len(order):
                issues.append(f'{field} length mismatch at {frame["t"]}')
        points = np.asarray([p if p is not None else [np.nan, np.nan] for p in frame['xy']], float)
        present = np.array([p is not None for p in frame['xy']])
        finite = np.isfinite(points).all(1)
        measured = np.array([v is True for v in frame.get('obs', present.tolist())]) & present
        totals['present'] += int(present.sum())
        totals['measured'] += int(measured.sum())
        totals['estimated'] += int((present & ~measured).sum())
        totals['nonfinite'] += int((present & ~finite).sum())
        totals['outside_pitch'] += int((finite & ((points[:, 0] < 0) | (points[:, 0] > 105) | (points[:, 1] < 0) | (points[:, 1] > 68))).sum())
        if previous is not None:
            old_points, old_measured, old_time = previous
            both = measured & old_measured & finite & np.isfinite(old_points).all(1)
            dt = frame['t']-old_time
            if dt > 0:
                speeds = np.linalg.norm(points[both]-old_points[both], axis=1)/dt
                totals['measured_transitions'] += len(speeds)
                totals['measured_jump_over_10m_s'] += int((speeds > 10).sum())
        previous = points, measured, frame['t']
    if totals['nonfinite']:
        issues.append('Present positions contain non-finite coordinates')
    output = dict(file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), frames=len(times), slots=len(order),
                  source_type=meta.get('source', {}).get('type'), source=meta.get('source', {}),
                  date=video_details(meta)['date'] if meta.get('source', {}).get('type') == 'computer_vision' else meta.get('date'),
                  score_known=meta.get('score_known', True), totals=totals, contract_issues=issues)
    try:
        intelligence = MatchIntelligence(source=str(path))
        states = []
        for timestamp in sorted({0, int(meta['duration'])//2, int(meta['duration'])}):
            state = intelligence.state_at(timestamp)
            json.dumps(state, allow_nan=False)
            states.append(dict(t=timestamp, players=len(state['players']), coverage=state['coverage']))
        modes = {}
        for mode in ['replay', 'live', 'truth']:
            try:
                frames = intelligence.frames(mode)
                modes[mode] = dict(status='available', frames=len(frames))
            except ValueError as error:
                modes[mode] = dict(status='explicitly rejected', reason=str(error))
        summary = intelligence.summary()
        json.dumps(summary, allow_nan=False)
        output.update(smoke='passed', modes=modes, sampled_states=states, moments=len(intelligence.moments))
        del intelligence
    except Exception as error:
        output.update(smoke='failed', error=f'{type(error).__name__}: {error}')
    gc.collect()
    output['seconds'] = time.time()-started
    return output


def main():
    root = Path('cv/out/audit8_20261011/active_matches')
    root.mkdir(parents=True, exist_ok=True)
    paths = sorted([*Path('data').glob('dfl_*_h[12].json'), *[p for p in Path('data').glob('cv_*.json') if 'backup' not in p.name]])
    results = []
    for path in paths:
        result = audit(path)
        results.append(result)
        (root/'results.json').write_text(json.dumps(results, indent=2))
        print(path, result['smoke'], result.get('error', ''), round(result['seconds'], 2), flush=True)
    rows = []
    for result in results:
        values = [result['file'], result['source_type'], result['date'] or 'N/A', result['frames'], result['slots'],
                  f"{100*result['totals']['measured']/max(1,result['totals']['present']):.1f}%", result['totals']['measured_jump_over_10m_s'],
                  result['smoke'], '; '.join(result['contract_issues']) or 'None found']
        rows.append('<tr>'+''.join('<td>'+escape(str(value))+'</td>' for value in values)+'</tr>')
    head = ''.join('<th>'+label+'</th>' for label in ['Export', 'Source', 'Export date', 'Frames', 'Roster slots', 'Measured share', 'Measured jumps >10m/s', 'Smoke', 'Contract issues'])
    (root/'review.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Active export audit</title><style>body{font:15px system-ui;margin:24px}table{border-collapse:collapse}th,td{border:1px solid #ccc;padding:8px}th{background:#eee}</style><h1>Read-only active match audit</h1><p>Smoke execution is not accuracy validation. Roster slots can include substitutes. Measured jumps may reflect perception/identity errors, genuine tracking anomalies or timing; 10 m/s is a diagnostic threshold. Out-of-pitch truth points are not silently clipped. Dates here reflect exported provenance, not catalog XML metadata.</p><table><thead><tr>'+head+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table></html>')
    print('Completed', len(results), 'exports; failures', sum(r['smoke'] != 'passed' for r in results), flush=True)


if __name__ == '__main__':
    main()
