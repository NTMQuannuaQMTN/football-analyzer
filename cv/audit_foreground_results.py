"""Summarise frozen camera candidates without altering positions or active exports."""
import json
from pathlib import Path

import numpy as np

from cv.audit_anonymous_continuity import summary
from cv.audit_camera_uncertainty import fit_bins, evaluate
from cv.audit_visible_pipeline import stats
from cv.session_report import table


ROOT = Path('cv/out/audit8_20261011')


def transitions(rows):
    result = {}
    for before, after in zip(rows, rows[1:]):
        dt = after['t'] - before['t']
        if not before['live'] or not after['live'] or abs(dt - .2) > .001:
            continue
        old = {d['id']: d for d in before['dets'] if d['cls'] == 'person'}
        for current in after['dets']:
            previous = old.get(current['id']) if current['cls'] == 'person' else None
            if previous is None or previous.get('shot_id') != current.get('shot_id'):
                continue
            key = (before['t'], after['t'], current['id'], current.get('shot_id'))
            if key in result:
                raise ValueError('Duplicate anonymous transition')
            result[key] = float(np.hypot(current['px']-previous['px'], current['py']-previous['py'])/dt)
    return result


def main():
    names = {'Original': 'broadcast/appearance_raw', 'Temporal': 'camera_continuity/appearance_raw',
             'Foreground': 'camera_foreground_arsenal/appearance_raw'}
    sets = {}; counts = {}
    for name, path in names.items():
        rows = [json.loads(line) for line in (ROOT/path/'detections.jsonl').open()]
        sets[name] = transitions(rows)
        counts[name] = sum(d['cls'] == 'person' for r in rows for d in r['dets'])
    common = sorted(set.intersection(*(set(value) for value in sets.values())))
    results = {name: {'supported_people': counts[name], 'all': summary(list(value.values())),
                     'paired': summary([value[key] for key in common])} for name, value in sets.items()}
    body = ['<h1>Frozen foreground-camera results</h1><p>Private retrospective broadcast review. '
            'Identical detection cache and anonymous association; only camera placement changes. '
            'Speed flags measure discontinuity, not ground-truth accuracy. The common set excludes '
            'missing candidate measurements. Active matches remain unchanged.</p>']
    rows = []
    for name, value in results.items():
        for subset in ['all', 'paired']:
            r = value[subset]
            rows.append([name, subset, value['supported_people'], r['n'], round(r['median_m_s'], 3),
                         round(r['p90_m_s'], 3), r['steps_over_10m_s'], round(r['max_m_s'], 3)])
    body.append(table(['Camera', 'Transition set', 'Projected people', 'Transitions', 'Median m/s',
                       'P90 m/s', '>10 m/s flags', 'Maximum m/s'], rows))
    pipeline = {}; rows = []; paired = {}; calibration_records = {}
    for split in ['val', 'test']:
        pipeline[split] = json.loads((ROOT/'visible_foreground'/f'{split}.json').read_text())
        matched = {}
        for variant in ['selected/dense_strict/kalman', 'selected/dense_foreground/kalman']:
            r = pipeline[split]['results'][variant]; e = r['position_errors_on_matched_supported_people']
            rows.append([split, variant, r['gt_total'], e['n'], round(e['mean_m'], 3),
                         round(e['median_m'], 3), round(e['p90_m'], 3),
                         f"{r['within_5m_recall_all_labelled_people']:.2%}"])
            matched[variant] = {(r['clip'], r['image'], r['gt_track']): r['error_m']
                for r in pipeline[split]['rows'] if r['variant']==variant}
        strict, foreground = matched.values()
        common_people = sorted(set(strict)&set(foreground))
        paired[split] = {'strict': stats([strict[key] for key in common_people]),
                         'foreground': stats([foreground[key] for key in common_people]),
                         'conditions': 'Exact same clip, image and reference identity; IoU matching held fixed.'}
        calibration_records[split] = [dict(bin='all', errors_m=[row['error_m']], role=row['role'])
            for row in pipeline[split]['rows'] if row['variant']=='selected/dense_foreground/kalman']
    body += ['<h2>Actual detected and tracked visible placement</h2><p>Fixed cached detector, '
             'selected BOTSORT, filtered feet, all 5 Hz clip frames. Validation: eight clips; '
             'test: eighteen clips from one held-out game. All-person denominator includes '
             'players, keepers, referees and missing pitch labels. Conditional metre metrics '
             'score only matched, supported, finite pitch references; compared sets differ. '
             'Camera-only common-reference results are reported separately in the audit.</p>',
             table(['Split', 'Camera / tracker', 'All GT people', 'Scored matches', 'Mean m',
                    'Median m', 'P90 m', 'Within 5 m / all GT people'], rows),
             '<p><a href="jumps/review.html">Ten symptom-selected jump cases</a> · '
             '<a href="foreground_8s.mp4">Private eight-second foreground movie</a></p>']
    calibration = fit_bins(calibration_records['val'])
    uncertainty = {'calibration': calibration, 'validation': evaluate(calibration_records['val'], calibration),
                   'test': evaluate(calibration_records['test'], calibration)}
    rows = []
    for split, value in paired.items():
        for camera in ['strict', 'foreground']:
            e = value[camera]
            rows.append([split, camera, e['n'], round(e['mean_m'], 3), round(e['median_m'], 3), round(e['p90_m'], 3)])
    body += ['<h2>Same supported detections</h2><p>These conditional comparisons hold the reference-person set fixed. '
             'They exclude new support and misses; they do not measure all-player accuracy.</p>',
             table(['Split', 'Camera', 'Paired matches', 'Mean m', 'Median m', 'P90 m'], rows)]
    rows = []
    for split in ['validation', 'test']:
        e = uncertainty[split]; c = e['bins']['all']
        rows.append([split, e['n'], round(c['radius_80_m'], 3), f"{e['radius_coverage']:.2%}",
                     f"{c['predicted_probability_within_5m']:.2%}", f"{c['observed_fraction_within_5m']:.2%}"])
    body += ['<h2>Validation-fitted visible-position radius</h2><p>One pooled empirical marginal radius fitted '
             'only on validation supported IoU matches. Missing detections, wrong named identities and unsupported '
             'cameras are excluded. Correlated clips and domain shift prevent a confidence guarantee; this is '
             'not transferred to the EPL app.</p>', table(['Split', 'Matched people', 'Nominal-80% radius m',
                 'Actual radius coverage', 'Predicted within 5 m', 'Observed within 5 m'], rows)]
    output = ROOT/'camera_foreground_arsenal'
    (output/'comparison.json').write_text(json.dumps({'broadcast': results, 'paired_n': len(common),
        'visible_pipeline_paired': paired, 'visible_uncertainty': uncertainty}, indent=2))
    (output/'review.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<title>Foreground camera comparison</title><style>body{font:16px system-ui;max-width:1250px;'
        'margin:30px auto;padding:20px}table{border-collapse:collapse;width:100%}'
        'td,th{border:1px solid #aaa;padding:8px}</style>'+''.join(body)+'</html>')
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
