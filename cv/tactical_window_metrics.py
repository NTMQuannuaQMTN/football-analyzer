"""Retrospective scoring of the application's minute/window geometry changes."""
import numpy as np


def comparison_windows(reference,estimated,possession,team):
    """Match raw-minute means and five-minute sides used by shift detection.

    Both arrays contain [width, depth, defensive line] in own-team metres.
    This scorer deliberately uses subsequent windows; it is never a live input.
    Defensive-line samples are out of possession, as in window_metrics().
    """
    reference=np.asarray(reference,float);estimated=np.asarray(estimated,float)
    possession=np.asarray(possession)
    if reference.shape!=estimated.shape or reference.ndim!=2 or reference.shape[1]!=3 or len(possession)!=len(reference):
        raise ValueError('Geometry arrays and possession must share the frame clock')
    names=['width_m','depth_m','def_line_oop_m'];minutes=[]
    for start in range(0,len(reference),60):
        end=min(start+60,len(reference));row=[]
        for metric in range(3):
            eligible=np.isfinite(reference[start:end,metric])&np.isfinite(estimated[start:end,metric])
            if metric==2:eligible&=(possession[start:end]>=0)&(possession[start:end]!=(0 if team=='A' else 1))
            row.append([round(float(array[start:end,metric][eligible].mean()),2) if eligible.any() else np.nan
                        for array in [reference,estimated]])
        minutes.append(row)
    minutes=np.asarray(minutes);output={key:[] for key in names}
    for minute in range(3,len(minutes)-2):
        for metric,name in enumerate(names):
            before=minutes[max(0,minute-5):minute,metric];after=minutes[minute:minute+5,metric]
            valid_before=np.isfinite(before).all(1);valid_after=np.isfinite(after).all(1)
            if not valid_before.any() or not valid_after.any():continue
            # window_metrics rounds each side to one decimal before taking a delta.
            left=np.round(before[valid_before].mean(0),1);right=np.round(after[valid_after].mean(0),1)
            delta=np.round(right-left,1)
            output[name].append(dict(minute=minute,reference_delta_m=float(delta[0]),estimated_delta_m=float(delta[1]),
                before_usable_minutes=int(valid_before.sum()),after_usable_minutes=int(valid_after.sum())))
    return output


def summarize_changes(rows,threshold=2):
    reference=np.array([r['reference_delta_m'] for r in rows]);estimated=np.array([r['estimated_delta_m'] for r in rows])
    true_change=np.abs(reference)>=threshold;claimed=np.abs(estimated)>=threshold
    correct=claimed&true_change&(reference*estimated>0)
    error=np.abs(reference-estimated)
    return dict(n=len(rows),mean_absolute_delta_error_m=float(error.mean()) if len(rows) else None,
        median_absolute_delta_error_m=float(np.median(error)) if len(rows) else None,
        p90_absolute_delta_error_m=float(np.percentile(error,90)) if len(rows) else None,
        estimated_changes=int(claimed.sum()),reference_changes=int(true_change.sum()),
        same_direction_reference_changes=int(correct.sum()),unsupported_threshold_changes=int((claimed&~true_change).sum()),
        opposite_direction_changes=int((claimed&true_change&~correct).sum()),
        missed_reference_changes=int((true_change&~claimed).sum()),
        same_direction_fraction_of_estimated_changes=float(correct.sum()/claimed.sum()) if claimed.any() else None,
        condition='Overlapping retrospective windows; 2m geometric threshold is not a tactical-intent label or significance test')
