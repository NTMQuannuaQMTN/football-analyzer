"""Empirical camera-error calibration on validation, scoring held-out coverage.

Scope is exact visible reference feet under the evaluated camera model. These
probabilities do not include detection, named identity or hidden-player error,
and must not be transferred to the Arsenal application as confidence.
"""
import json
from pathlib import Path

import cv2
import numpy as np

from cv.camera_quality import assess_camera


def fit_bins(records):
    groups={}
    for r in records:groups.setdefault(r['bin'],[]).extend(r['errors_m'])
    return {key:dict(n=len(values),radius_80_m=float(np.percentile(values,80)),
                    probability_within_5m=(sum(v<=5 for v in values)+1)/(len(values)+2)) for key,values in groups.items() if len(values)>=100}


def evaluate(records,bins):
    results={};all_hit=[];all_probability=[];all_truth=[]
    for name,cal in bins.items():
        errors=np.array([e for r in records if r['bin']==name for e in r['errors_m']],float)
        if not len(errors):continue
        truth=errors<=5;hit=errors<=cal['radius_80_m'];prob=cal['probability_within_5m']
        results[name]=dict(n=len(errors),radius_80_m=cal['radius_80_m'],radius_coverage=float(hit.mean()),
            predicted_probability_within_5m=prob,observed_fraction_within_5m=float(truth.mean()),mean_error_m=float(errors.mean()))
        all_hit.extend(hit.tolist());all_probability.extend([prob]*len(errors));all_truth.extend(truth.tolist())
    p=np.array(all_probability);y=np.array(all_truth,float)
    return dict(bins=results,n=len(y),radius_coverage=float(np.mean(all_hit)) if len(y) else None,
        brier_score=float(np.mean((p-y)**2)) if len(y) else None,
        weighted_bin_calibration_error=sum(r['n']*abs(r['predicted_probability_within_5m']-r['observed_fraction_within_5m']) for r in results.values())/max(1,len(y)),
        unsupported_calibration_people=sum(len(r['errors_m']) for r in records if r['bin'] not in bins))


def records(source,configuration):
    out=[]
    for r in source['cases']:
        if r['configuration']!=configuration or not r['supported']:continue
        quality=assess_camera(cv2.imread(r['image']),np.array(r['H']))
        scale=quality.get('p90_metres_per_original_pixel',float('inf'))
        bucket='small' if scale<=.05 else 'medium' if scale<=.1 else 'large'
        kind='anchor' if r['anchor_age_s']<.01 else 'propagated'
        out.append(dict(bin=kind+'_'+bucket,clip=r['clip'],image=r['image'],errors_m=r['errors_m']))
    return out


def main():
    root=Path('cv/out/audit8_20261011/camera_motion');val=json.loads((root/'val.json').read_text());test=json.loads((root/'test.json').read_text())
    selection=json.loads((root/'selection.json').read_text())
    validation=records(val,selection['name']);heldout=records(test,'selected')
    calibration=fit_bins(validation);result=dict(calibration=calibration,validation=evaluate(validation,calibration),test=evaluate(heldout,calibration),
        conditions='Validation empirical marginal position radii and smoothed <=5m frequencies, grouped by image-only projective scale and propagation age. Held-out game evaluates coverage. Exact labelled visible feet are used as image inputs; boxes/detector, named identity and hidden position are excluded. Samples within a clip are correlated. No conformal guarantee, no EPL-domain transfer, no production confidence change.')
    (root/'uncertainty.json').write_text(json.dumps(result,indent=2));print(result,flush=True)


if __name__=='__main__':main()
