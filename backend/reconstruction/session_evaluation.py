"""Exact-ID, held-out tracking evaluation; reference positions are scoring-only."""
import numpy as np

from backend.reconstruction import ml
from backend.reconstruction.datasets import L, W, parse_dfl
from backend.reconstruction.real_benchmark import formation, last_position, make_mask, summarize as _summary


def summarize(error):
    error=np.asarray(error)
    eligible=len(error)
    error=error[np.isfinite(error)]
    value = _summary(error)
    if value is not None:
        value.update(median_m=float(np.median(error)), p90_m=float(np.percentile(error,90)),
                     eligible_n=eligible,excluded_nonfinite_n=eligible-len(error))
    return value


def evaluate(net, calibration, mode, version, match):
    info, segs = parse_dfl(match)
    results = {}
    for mask in ("sc_camera", "k=14", "k=10", "k=6"):
        rows = {"model": [], "model_speed_limited": [], "formation": [], "last_position": []}
        coverage, radii, ages, roles, confidences, phases = [], [], [], [], [], []
        jumps, limited_jumps, limited_coverage = [], [], []
        shape_errors={"width_m":[],"depth_m":[],"defensive_line_m":[]}
        for si, seg in enumerate(segs):
            vis = make_mask(seg, mask, 777+si)
            on = np.isfinite(seg["pos"]).all(axis=2)
            hidden = on & ~vis
            prior = formation(info, seg, vis, mode).pos
            x, idx = ml.features(np.where(vis[...,None],seg["pos"],np.nan), vis,on,
                                seg["team"],seg["role"],seg["ball"],seg["poss"],
                                seg["alive"],prior,mode,version=version)
            pred = prior.copy()
            t,j = idx[:,0], idx[:,1]
            correction,sigma = ml.predict(net,x)
            pred[t,j] = np.clip(prior[t,j]+np.where((seg["team"][j]=="B")[:,None],-correction,correction),[0,0],[L,W])
            error = np.linalg.norm(pred[t,j]-seg["pos"][t,j],axis=1)
            radius,confidence = ml.apply_calibration(calibration,sigma)
            coverage.append(error<=radius)
            radii.append(radius)
            ages.append(ml.since_until(vis)[0][t,j])
            roles.append(seg["role"][j])
            confidences.append(confidence)
            phases.append(np.where(~seg["alive"][t],"dead_ball",np.where(seg["poss"][t]<0,"unknown",np.where(seg["poss"][t]==(seg["team"][j]=="B"),"in_possession","out_of_possession"))))
            from backend.reconstruction.postprocess import limit_speed
            limited = limit_speed(pred.copy(),vis,on,causal=mode=="live")
            from backend.tactics.position_evidence import geometry
            for tm in ("A","B"):
                js=np.where((seg["team"]==tm)&(seg["role"]!="GK"))[0]
                for second in range(len(vis)):
                    active=js[on[second,js]]
                    truth_points=seg["pos"][second,active].copy()
                    estimated=limited[second,active].copy()
                    if tm=="B":
                        truth_points=[L,W]-truth_points;estimated=[L,W]-estimated
                    reference_metrics=geometry(truth_points);estimated_metrics=geometry(estimated)
                    if reference_metrics and estimated_metrics:
                        for key in shape_errors:
                            shape_errors[key].append(abs(reference_metrics[key]-estimated_metrics[key]))
            for key,p in (("model",pred),("model_speed_limited",limited),("formation",prior),("last_position",last_position(seg,vis))):
                rows[key].append(np.linalg.norm(p-seg["pos"],axis=2)[hidden])
            transitions = hidden[1:] & hidden[:-1]
            jumps.append(np.linalg.norm(np.diff(pred,axis=0),axis=2)[transitions])
            limited_jumps.append(np.linalg.norm(np.diff(limited,axis=0),axis=2)[transitions])
            limited_coverage.append(np.linalg.norm(limited[t,j]-seg["pos"][t,j],axis=1)<=radius)
        errors = np.concatenate(rows["model"])
        age,role = np.concatenate(ages),np.concatenate(roles)
        results[mask] = {k:summarize(np.concatenate(v)) for k,v in rows.items()}
        last_error=np.concatenate(rows["last_position"])
        common=np.isfinite(last_error)&np.isfinite(errors)
        results[mask]["paired_last_observed"]={
            "model":summarize(errors[common]),"last_position":summarize(last_error[common]),
            "condition":"Identical hidden player-timestamps with a previous sighting; never-seen players excluded from both methods"}
        results[mask]["model"].update(radius_coverage_80=float(np.concatenate(coverage).mean()),
                                     mean_radius_m=float(np.concatenate(radii).mean()),
                                     hidden_steps_over_10m_s=int((np.concatenate(jumps)>10).sum()),
                                     hidden_transition_count=int(sum(len(v) for v in jumps)))
        results[mask]["model_speed_limited"].update(
            hidden_steps_over_10m_s=int((np.concatenate(limited_jumps)>10).sum()),
            hidden_transition_count=int(sum(len(v) for v in limited_jumps)),
            radius_coverage_80=float(np.concatenate(limited_coverage).mean()),
            condition="Raw correction followed by existing speed limiter; goalkeeper-zone and event-cue priors excluded from this controlled comparison")
        results[mask]["by_age"] = {name:summarize(errors[sel]) for name,sel in (
            ("0_to_3_s",age<=3),("3_to_10_s",(age>3)&(age<=10)),
            ("10_to_30_s",(age>10)&(age<=30)),("over_30_s",(age>30)&np.isfinite(age)),("never_seen",~np.isfinite(age)))}
        results[mask]["by_role"] = {str(r):summarize(errors[role==r]) for r in np.unique(role)}
        confidence,phase=np.concatenate(confidences),np.concatenate(phases)
        results[mask]["by_confidence"] = {name:summarize(errors[sel]) for name,sel in (
            ("low_p5_below_0.4",confidence<.4),("medium_p5_0.4_to_0.7",(confidence>=.4)&(confidence<.7)),("high_p5_at_least_0.7",confidence>=.7))}
        results[mask]["by_phase"] = {str(p):summarize(errors[phase==p]) for p in np.unique(phase)}
        results[mask]["tactical_geometry_absolute_error_m"]={k:summarize(v) for k,v in shape_errors.items()}
    return dict(match=match,mode=mode,feature_version=version,sampling_hz=1,
                identity="Exact tracking IDs; no video perception error included",results=results)
