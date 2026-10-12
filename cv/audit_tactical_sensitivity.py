"""Score position-derived tactical geometry after inference, never as model input."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from backend.reconstruction import ml
from backend.reconstruction.datasets import L, W, parse_dfl
from backend.reconstruction.estimator import reconstruct
from backend.reconstruction.inference import load_ml
from backend.reconstruction.real_benchmark import make_mask, segment_view
from backend.tactics.position_evidence import geometry, position_evidence
from cv.audit_hidden_sensitivity import corrupt, require_held_out
from cv.audit_visible_pipeline import stats
from cv.session_report import table
from cv.tactical_window_metrics import comparison_windows, summarize_changes
from cv.audit_ball_conditioning import context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--match', required=True)
    parser.add_argument('--live-checkpoint', type=Path, required=True)
    parser.add_argument('--replay-checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--window-changes', action='store_true', help='Also score retrospective five-minute geometry changes; never a live input')
    parser.add_argument('--ball-context-stress', action='store_true', help='Exact visible positions; compare exact context with past-only ball holds every30s')
    args = parser.parse_args()
    torch.set_num_threads(1)
    info, segments = parse_dfl(args.match)
    conditions = ['exact_context','ball_observed_every_30s'] if args.ball_context_stress else ['clean_visible', 'gaussian_visible_mean_2m', 'camera_shift_5m_x']
    results = {}; rows = []; window_results = {}; window_rows = []
    for mode in ['live', 'replay']:
        checkpoint = getattr(args, f'{mode}_checkpoint')
        require_held_out(torch.load(checkpoint, weights_only=False), args.match)
        net, calibration, version = load_ml(mode, checkpoint.stem.removeprefix(f'hidden_{mode}_'), checkpoint.parent)
        values = {condition: {metric: {'error': [], 'covered': [], 'span': []}
                              for metric in ['width_m', 'depth_m', 'defensive_line_m']}
                  for condition in conditions}
        windows = {condition: {metric: [] for metric in ['width_m','depth_m','def_line_oop_m']} for condition in conditions}
        for segment_index, segment in enumerate(segments):
            visible = make_mask(segment, 'sc_camera', 777+segment_index)
            on = np.isfinite(segment['pos']).all(2)
            for condition in conditions:
                if args.ball_context_stress:
                    measurements=np.where(visible[...,None],segment['pos'],np.nan)
                    available=visible
                    ball,possession=context(segment['ball'],segment['poss'],condition)
                else:
                    measurements, available, ball = corrupt(segment, visible, condition, 11026+segment_index)
                    possession=segment['poss']
                view, _, roles, anchors = segment_view(info, segment)
                view.xy = measurements; view.ball = ball; view.poss=possession
                prior = reconstruct(view, available, mode, {}, on=on, role=roles, anchors=anchors).pos
                features, indices = ml.features(measurements, available, on, segment['team'], segment['role'],
                    ball, possession, segment['alive'], prior, mode, version=version)
                correction, sigma = ml.predict(net, features)
                predicted = prior.copy(); t, j = indices[:, 0], indices[:, 1]
                predicted[t, j] = np.clip(prior[t, j] + np.where((segment['team'][j]=='B')[:, None],
                    -correction, correction), [0, 0], [L, W])
                radius, _ = ml.apply_calibration(calibration, sigma)
                radii = np.full(available.shape, np.nan); radii[t, j] = radius
                frame_geometry = {team: {name: np.full((len(available),3),np.nan) for name in ['reference','estimated']}
                                  for team in ['A','B']}
                for second in range(len(available)):
                    for team in ['A', 'B']:
                        active = (segment['team']==team)&(segment['role']!='GK')&on[second]
                        own = predicted[second].copy()
                        if team=='B': own = [L, W]-own
                        frame = dict(t=second, obs=available[second].tolist(), rad=radii[second].tolist())
                        match = SimpleNamespace(T=1, order=list(range(len(active))), frames=[frame],
                            team_mask=lambda ignored, mask=active: mask,
                            own_xy=lambda ignored, positions=own: positions[None])
                        evidence = position_evidence(match, team, 0)
                        reference = segment['pos'][second, active].copy()
                        if team=='B': reference = [L, W]-reference
                        truth = geometry(reference)
                        estimate = geometry(own[active])
                        if truth is not None and estimate is not None:
                            for metric_index,metric_name in enumerate(['width_m','depth_m','defensive_line_m']):
                                frame_geometry[team]['reference'][second,metric_index]=truth[metric_name]
                                frame_geometry[team]['estimated'][second,metric_index]=estimate[metric_name]
                        if not (~available[second,active]).any():continue
                        if truth is None or estimate is None or evidence['sensitivity_bounds_m'] is None:
                            continue
                        for metric, reference_value in truth.items():
                            record = values[condition][metric]
                            low, high = evidence['sensitivity_bounds_m'][metric]
                            record['error'].append(abs(estimate[metric]-reference_value))
                            record['covered'].append(low <= reference_value <= high)
                            record['span'].append(high-low)
                if args.window_changes:
                    for team,arrays in frame_geometry.items():
                        changes=comparison_windows(arrays['reference'],arrays['estimated'],segment['poss'],team)
                        for metric,records in changes.items():
                            windows[condition][metric].extend([dict(segment=segment_index,team=team,**record) for record in records])
                print(mode, segment_index, condition, flush=True)
        results[mode] = {'checkpoint': str(checkpoint), 'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(), 'conditions': {}}
        for condition, metrics in values.items():
            results[mode]['conditions'][condition] = {}
            for metric, record in metrics.items():
                summary = stats(record['error'])
                summary.update(sensitivity_envelope_coverage=float(np.mean(record['covered'])),
                               mean_envelope_span_m=float(np.mean(record['span'])))
                results[mode]['conditions'][condition][metric] = summary
                rows.append([mode, condition, metric, summary['n'], round(summary['mean_m'], 3),
                    round(summary['median_m'], 3), round(summary['p90_m'], 3),
                    f"{summary['sensitivity_envelope_coverage']:.2%}", round(summary['mean_envelope_span_m'], 2)])
        if args.window_changes:
            window_results[mode]={}
            for condition,metrics in windows.items():
                window_results[mode][condition]={}
                for metric,records in metrics.items():
                    value=summarize_changes(records);window_results[mode][condition][metric]=dict(summary=value,rows=records)
                    window_rows.append([mode,condition,metric,value['n'],round(value['mean_absolute_delta_error_m'],3),
                        round(value['p90_absolute_delta_error_m'],3),value['estimated_changes'],value['reference_changes'],
                        value['same_direction_reference_changes'],value['unsupported_threshold_changes'],value['opposite_direction_changes'],value['missed_reference_changes']])
    explanation = ('One held-out DFL match, both halves, fixed simulated camera masks and exact identities/context. '
        'Frame-team samples require at least seven active outfield positions and at least one inferred player. '
        'Truth enters scoring only after inference. Raw model estimates, without a speed limiter or event cues. '
        'Production geometry/radius-envelope calculations are reused. Measured-player radii are zero, so '
        'perception errors are deliberately not included in those envelopes. Hidden radii are marginal '
        'validation-fitted quantities; geometric bounds are sensitivity envelopes, not simultaneous confidence '
        'intervals. No training or recalibration; no tactical intent or causal outcome is scored. '
        'Live inputs are past/current only; replay may use future visible measurements.')
    args.out.mkdir(parents=True, exist_ok=True)
    if args.ball_context_stress:
        explanation+=' Context-only comparison: visible positions, identities, possession and match phase remain exact. Stale ball uses the last exact past/current sample every30 seconds; no future ball information. These are controlled conditions, not measured EPL ball errors.'
    if args.window_changes:
        explanation+=' Additional geometry-change scoring uses retrospective five-minute sides, raw-minute means, out-of-possession defensive line, application rounding and fixed 2m thresholds. Overlapping windows are not independent tactical events; no intention or event outcome labels are supplied.'
    (args.out/'results.json').write_text(json.dumps({'match': args.match, 'seed': 11026, 'conditions': explanation,
        'results': results,'window_changes':window_results,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'window_source_sha256':hashlib.sha256(Path('cv/tactical_window_metrics.py').read_bytes()).hexdigest()}, indent=2))
    (args.out/'review.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<title>Tactical geometry sensitivity</title><style>body{font:15px system-ui;max-width:1300px;'
        'margin:30px auto;padding:16px}table{border-collapse:collapse;width:100%}'
        'th,td{border:1px solid #aaa;padding:7px}</style><h1>Tactical geometry under position error</h1><p>'
        + explanation + '</p>' + table(['Mode', 'Visible input', 'Feature', 'Frame-team N', 'Mean absolute error m',
            'Median m', 'P90 m', 'Reference inside sensitivity envelope', 'Mean envelope span m'], rows)
        +('<h2>Retrospective geometry-change thresholds</h2><p>Raw-frame error must not be confused with window-delta error: '
          'a constant shared camera bias can cancel between windows. Threshold agreement scores geometry only. '
          'Subsequent windows enter evaluation; they never enter live prediction. Sensitivity bounds at one frame '
          'do not establish confidence for a window change.</p>'+table(['Mode','Visible input','Feature','Window-team N',
            'Mean delta error m','P90 delta error m','Estimated ≥2m changes','Reference ≥2m changes','Same-direction matches',
            'Below-threshold reference','Opposite direction','Missed reference changes'],window_rows) if args.window_changes else '')+'</html>')


if __name__ == '__main__':
    main()
