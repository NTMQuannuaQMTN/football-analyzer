"""Validation-fit radii for matched detected feet, with explicit missing coverage.

This is an evaluation artifact, not a confidence model for EPL or named players.
"""
import json
from pathlib import Path

import numpy as np

from cv.audit_camera_uncertainty import fit_bins, evaluate
from cv.audit_visible_pipeline import stats


VARIANT = 'selected/learned_propagated/kalman'


def records(root, part):
    pipeline = json.loads((root / 'visible_pipeline' / f'{part}.json').read_text())
    motion = json.loads((root / 'camera_motion' / f'{part}.json').read_text())
    configuration = json.loads((root / 'camera_motion/selection.json').read_text())['name'] if part == 'val' else 'selected'
    cameras = {(r['clip'], Path(r['image']).stem): r for r in motion['cases'] if r['configuration'] == configuration}
    output = []
    for row in pipeline['rows']:
        if row['variant'] != VARIANT:
            continue
        camera = cameras[row['clip'], Path(row['image']).stem]
        kind = 'anchor' if camera['anchor_age_s'] < .01 else 'propagated'
        output.append(dict(bin=kind, errors_m=[row['error_m']], role=row['role'], clip=row['clip']))
    return output, pipeline['results'][VARIANT]


def main():
    root = Path('cv/out/audit8_20261011')
    validation, val_totals = records(root, 'val')
    heldout, test_totals = records(root, 'test')
    calibration = fit_bins(validation)
    grouped = {}
    for part, rows in [('val', validation), ('test', heldout)]:
        grouped[part] = {role: stats([r['errors_m'][0] for r in rows if r['role'] == role]) for role in sorted({r['role'] for r in rows})}
    output = dict(variant=VARIANT, calibration=calibration,
                  validation=evaluate(validation, calibration), test=evaluate(heldout, calibration),
                  visibility_denominators=dict(validation=val_totals, test=test_totals), role_errors=grouped,
                  conditions='Same frozen detection/tracker/camera pipeline. Reference labels match and score only; predictions use images. Radii fit validation matched supported position errors; missing detections/cameras and wrong named identities are outside conditional calibration. No hidden-player, EPL-domain or conformal guarantee. All test clips share one game.')
    (root / 'visible_pipeline/uncertainty.json').write_text(json.dumps(output, indent=2))
    print(output, flush=True)


if __name__ == '__main__':
    main()
