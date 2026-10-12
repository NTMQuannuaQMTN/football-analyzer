"""Controlled perception-error stress test, not a real-video accuracy estimate.

Uses a previously validation-selected, match-held-out model without retraining.
Hidden reference coordinates score only. Live inputs are past/current visible
measurements; replay may use future visible measurements explicitly.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from backend.reconstruction import ml
from backend.reconstruction.datasets import L, W, parse_dfl
from backend.reconstruction.estimator import reconstruct
from backend.reconstruction.inference import load_ml
from backend.reconstruction.real_benchmark import make_mask, segment_view, last_position
from backend.reconstruction.session_evaluation import summarize


CONDITIONS = ['clean_visible', 'gaussian_visible_mean_2m', 'camera_shift_5m_x', 'identity_swap_10pct_frames', 'measurement_delay_1s', 'ball_shift_5m_x']


def require_held_out(checkpoint_metadata,match):
    if not checkpoint_metadata.get('train') or match in checkpoint_metadata['train'] or checkpoint_metadata.get('test')!=match:
        raise ValueError('Checkpoint must record this match as held out and absent from training')


def corrupt(seg, visible, name, seed):
    rng = np.random.default_rng(seed)
    measurements = np.where(visible[..., None], seg['pos'], np.nan).copy()
    available = visible.copy()
    ball = seg['ball'].copy()
    if name == 'gaussian_visible_mean_2m':
        measurements[visible] += rng.normal(0, 2/np.sqrt(np.pi/2), (visible.sum(), 2))
    elif name == 'camera_shift_5m_x':
        measurements[visible] += [5, 0]
    elif name == 'identity_swap_10pct_frames':
        for time_index in range(len(visible)):
            if rng.random() >= .1:
                continue
            for team in ['A', 'B']:
                candidates = np.flatnonzero(visible[time_index] & (seg['team'] == team) & (seg['role'] != 'GK'))
                if len(candidates) >= 2:
                    left, right = rng.choice(candidates, 2, replace=False)
                    measurements[time_index, [left, right]] = measurements[time_index, [right, left]]
    elif name == 'measurement_delay_1s':
        measurements[1:] = measurements[:-1].copy()
        measurements[0] = np.nan
        available[1:] = available[:-1].copy()
        available[0] = False
        # Substituted/off-pitch players cannot remain available after leaving.
        available &= np.isfinite(seg['pos']).all(2)
        measurements[~available] = np.nan
    elif name == 'ball_shift_5m_x':
        ball += [5, 0]
    elif name != 'clean_visible':
        raise ValueError(name)
    assert not np.isfinite(measurements[~available]).any(), 'Unavailable positions entered inference'
    return measurements, available, ball


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--live-checkpoint')
    parser.add_argument('--replay-checkpoint')
    parser.add_argument('--match',help='Other held-out match; requires explicit matching checkpoints with training provenance')
    parser.add_argument('--out',default='cv/out/audit8_20261011/hidden_sensitivity')
    args=parser.parse_args()
    if bool(args.live_checkpoint)!=bool(args.replay_checkpoint):
        raise ValueError('Supply both explicitly selected live/replay checkpoints')
    if args.match and not args.live_checkpoint:raise ValueError('Match override requires explicit held-out checkpoints')
    torch.set_num_threads(1)
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    training = Path('cv/out/training_20261011/reconstruction')
    selection = {} if args.match and args.live_checkpoint else json.loads((training/'results.json').read_text())
    match = args.match or selection['test']
    info, segments = parse_dfl(match)
    results = {}
    for mode in ['live', 'replay']:
        explicit=getattr(args,f'{mode}_checkpoint')
        checkpoint = Path(explicit) if explicit else training/selection['selected_test'][mode]['checkpoint']
        require_held_out(torch.load(checkpoint,weights_only=False),match)
        tag = checkpoint.stem.removeprefix(f'hidden_{mode}_')
        net, calibration, version = load_ml(mode, tag, checkpoint.parent)
        values = {condition: {'model': [], 'formation': [], 'last_observed': [], 'radius_hit': [], 'age': [], 'input_error': []} for condition in CONDITIONS}
        for segment_index, segment in enumerate(segments):
            visible = make_mask(segment, 'sc_camera', 777+segment_index)
            on = np.isfinite(segment['pos']).all(2)
            inputs = {condition: corrupt(segment, visible, condition, 11026+segment_index) for condition in CONDITIONS}
            common = on.copy()
            for _, available, _ in inputs.values():
                common &= ~available
            for condition, (measurement, available, ball) in inputs.items():
                view, _, roles, anchors = segment_view(info, segment)
                view.xy = measurement
                view.ball = ball
                prior = reconstruct(view, available, mode, {}, on=on, role=roles, anchors=anchors).pos
                features, indices = ml.features(measurement, available, on, segment['team'], segment['role'], ball,
                                                 segment['poss'], segment['alive'], prior, mode, version=version)
                correction, sigma = ml.predict(net, features)
                predicted = prior.copy()
                time_index, player = indices[:, 0], indices[:, 1]
                predicted[time_index, player] = np.clip(prior[time_index, player]+np.where((segment['team'][player] == 'B')[:, None], -correction, correction), [0, 0], [L, W])
                selected = common[time_index, player]
                error = np.linalg.norm(predicted[time_index, player]-segment['pos'][time_index, player], axis=1)[selected]
                radius, _ = ml.apply_calibration(calibration, sigma)
                values[condition]['model'].append(error)
                values[condition]['formation'].append(np.linalg.norm(prior-segment['pos'], axis=2)[common])
                last = last_position(dict(segment, pos=measurement), available)
                values[condition]['last_observed'].append(np.linalg.norm(last-segment['pos'], axis=2)[common])
                values[condition]['radius_hit'].append(error <= radius[selected])
                values[condition]['age'].append(ml.since_until(available)[0][time_index, player][selected])
                valid = available & on
                values[condition]['input_error'].append(np.linalg.norm(measurement-segment['pos'], axis=2)[valid])
                print(mode, segment_index, condition, len(error), flush=True)
        output = {}
        for condition, records in values.items():
            errors = np.concatenate(records['model'])
            ages = np.concatenate(records['age'])
            last = np.concatenate(records['last_observed'])
            has_last = np.isfinite(last)
            output[condition] = dict(model=summarize(errors), formation=summarize(np.concatenate(records['formation'])),
                last_observed=summarize(last), paired_previous_sighting=dict(model=summarize(errors[has_last]), last_observed=summarize(last[has_last])),
                radius_80_coverage=float(np.concatenate(records['radius_hit']).mean()), visible_input_error=summarize(np.concatenate(records['input_error'])),
                by_age={name:summarize(errors[mask]) for name, mask in [('up_to_3s', ages <= 3), ('3_to_10s', (ages > 3)&(ages <= 10)),
                    ('10_to_30s', (ages > 10)&(ages <= 30)), ('over_30s', (ages > 30)&np.isfinite(ages)), ('never_seen', ~np.isfinite(ages))]})
        results[mode] = dict(checkpoint=str(checkpoint), sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), feature_version=version, conditions=output)
        (root/'results.json').write_text(json.dumps(dict(match=match, seed=11026, results=results,
            conditions='Previously validation-selected model; one held-out DFL match, two halves, fixed simulated broadcast visibility. Synthetic corruptions are controlled stresses, not fitted EPL perception distributions. Exactly the same common hidden player-timestamps across conditions within each mode. Exact IDs/roles, known roster/ball/possession/phase except stated corruptions. No test training, recalibration, future hidden coordinates or model promotion. Replay is retrospective; live uses past/current available measurements. Last-observed full denominator includes explicitly counted never-seen exclusions.'), indent=2))


if __name__ == '__main__':
    main()
