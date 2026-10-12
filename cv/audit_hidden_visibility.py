"""Separate geometric off-camera players from missed in-camera observations.

Controlled tracking evaluation only; simulated reference visibility is not
available for real EPL footage. Never pushes a prediction outside a polygon.
"""
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import torch

from backend.reconstruction import ml
from backend.reconstruction.datasets import L, W, parse_dfl, point_in_quad
from backend.reconstruction.estimator import reconstruct
from backend.reconstruction.inference import load_ml
from backend.reconstruction.real_benchmark import mask_sc, make_mask, segment_view
from backend.reconstruction.session_evaluation import summarize
from cv.session_report import table


ROOT = Path('cv/out/audit8_20261011')


def footprint_membership(positions, quads):
    valid = np.isfinite(quads).all((1, 2))
    inside = np.zeros(positions.shape[:2], bool)
    for t in np.flatnonzero(valid):
        inside[t] = point_in_quad(positions[t], quads[t]) & np.isfinite(positions[t]).all(1)
    return inside, valid


def main():
    torch.set_num_threads(1)
    output = ROOT / 'hidden_visibility'
    output.mkdir(parents=True, exist_ok=True)
    selection = json.loads(Path('cv/out/training_20261011/reconstruction/results.json').read_text())
    match = selection['test']; info, segments = parse_dfl(match)
    models = {}
    for mode in ('live', 'replay'):
        paths = {'baseline': Path('cv/out/training_20261011/reconstruction') / selection['selected_test'][mode]['checkpoint'],
                 'robust': ROOT / f'hidden_robustness/hidden_{mode}_robustness_robust_v4.pt'}
        for name, checkpoint in paths.items():
            tag = checkpoint.stem.removeprefix(f'hidden_{mode}_')
            net, calibration, version = load_ml(mode, tag, checkpoint.parent)
            groups = {k: {'errors': [], 'radius_hits': [], 'predicted_inside': []}
                      for k in ('geometrically_outside', 'inside_but_not_observed', 'camera_unavailable')}
            for half, seg in enumerate(segments):
                visible, quads = mask_sc(seg, 777 + half, return_quads=True)
                np.testing.assert_equal(visible, make_mask(seg, 'sc_camera', 777 + half))
                on = np.isfinite(seg['pos']).all(2)
                measured = np.where(visible[..., None], seg['pos'], np.nan)
                view, _, roles, anchors = segment_view(info, seg); view.xy = measured
                prior = reconstruct(view, visible, mode, {}, on=on, role=roles, anchors=anchors).pos
                features, indices = ml.features(measured, visible, on, seg['team'], seg['role'], seg['ball'],
                                               seg['poss'], seg['alive'], prior, mode, version=version)
                correction, sigma = ml.predict(net, features)
                predicted = prior.copy(); t, j = indices[:, 0], indices[:, 1]
                predicted[t, j] = np.clip(prior[t, j] + np.where((seg['team'][j] == 'B')[:, None], -correction, correction), [0, 0], [L, W])
                reference_inside, camera_valid = footprint_membership(seg['pos'], quads)
                estimated_inside, _ = footprint_membership(predicted, quads)
                # Geometry is read here only for evaluation grouping, never features/prior.
                errors = np.linalg.norm(predicted[t, j] - seg['pos'][t, j], axis=1)
                radius, _ = ml.apply_calibration(calibration, sigma)
                selectors = {'geometrically_outside': camera_valid[t] & ~reference_inside[t, j],
                             'inside_but_not_observed': camera_valid[t] & reference_inside[t, j],
                             'camera_unavailable': ~camera_valid[t]}
                assert sum(int(s.sum()) for s in selectors.values()) == len(indices)
                for group, selected in selectors.items():
                    groups[group]['errors'].append(errors[selected])
                    groups[group]['radius_hits'].append((errors <= radius)[selected])
                    if group != 'camera_unavailable':
                        groups[group]['predicted_inside'].append(estimated_inside[t, j][selected])
                print(mode, name, half, len(indices), flush=True)
            summary = {}
            for group, records in groups.items():
                errors = np.concatenate(records['errors'])
                summary[group] = dict(error=summarize(errors), radius_80_coverage=float(np.concatenate(records['radius_hits']).mean()) if len(errors) else None,
                    predicted_inside_camera_fraction=float(np.concatenate(records['predicted_inside']).mean())
                    if records['predicted_inside'] and sum(map(len, records['predicted_inside'])) else None)
            models[f'{mode}/{name}'] = dict(checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), groups=summary)
            (output / 'results.json').write_text(json.dumps(dict(match=match, seed=777, models=models,
                conditions='One held-out DFL match, both halves, fixed transferred SkillCorner polygons and detection rates. Exact IDs/roles/context and visible reference feet, simulated visibility. These are not independent broadcast or EPL truth. Camera polygons and hidden reference positions classify evaluation groups only, never model inputs. Live uses past/current observations; replay can use future visible observations. No outside-polygon projection, hiding, test fitting or model promotion.'), indent=2))
    rows = []
    for name, model in models.items():
        for group, r in model['groups'].items():
            e = r['error']; fraction = r['predicted_inside_camera_fraction']
            rows.append([name, group, e['n'] if e else 0, e['mae_m'] if e else 'N/A',
                round(e['median_m'], 2) if e else 'N/A', round(e['p90_m'], 2) if e else 'N/A',
                f"{fraction:.2%}" if fraction is not None else 'N/A',
                f"{r['radius_80_coverage']:.2%}" if r['radius_80_coverage'] is not None else 'N/A'])
    conditions = json.loads((output / 'results.json').read_text())['conditions']
    (output / 'review.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Off-camera versus undetected</title><style>body{font:16px system-ui;max-width:1250px;margin:30px auto;padding:16px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #aaa;padding:8px}</style><h1>Off-camera is different from undetected</h1><p>' + html.escape(conditions) + '</p>' +
        table(['Mode / model','True visibility group','Hidden N','Mean m','Median m','P90 m','Prediction inside camera','Nominal-80% coverage'], rows) +
        '<p>A geometrically hidden player predicted inside the camera can indicate a positional error. '
        'An undetected player truly inside the camera should not be forced outside it. Neither '
        'case can be identified confidently in Arsenal without independent calibration/identity labels.</p></html>')
    print(output / 'review.html', flush=True)


if __name__ == '__main__':
    main()
