"""Position-derived evidence and conditional mechanisms, without invented intent."""
import numpy as np


def geometry(points):
    """Outfield metrics in own-team metres, consistent with frame_shape."""
    p=np.asarray(points,float)
    if len(p)<7 or not np.isfinite(p).all():
        return None
    return dict(width_m=float(np.ptp(p[:,1])),depth_m=float(np.ptp(p[:,0])),
                defensive_line_m=float(np.sort(p[:,0])[:4].mean()))


def position_evidence(m,team,t):
    t=int(np.clip(t,0,m.T-1))
    frame=m.frames[t]
    js=np.where(m.team_mask(team))[0]
    own=m.own_xy(team)[t,js]
    finite=np.isfinite(own).all(axis=1)
    own=own[finite]; js=js[finite]
    point=geometry(own)
    if point is None:
        return dict(status="insufficient_positions",metrics=None)
    obs=frame.get("obs") or [True]*len(m.order)
    inferred=[j for j in js if obs[j] is not True]
    radii=frame.get("rad") or [None]*len(m.order)
    known=all(radii[j] is not None and np.isfinite(radii[j]) and radii[j]>=0 for j in inferred)
    bounds=None
    if known:
        r=np.array([0 if obs[j] is True else radii[j] for j in js])
        # Box envelope of per-player circles. These are sensitivity bounds,
        # not a calibrated simultaneous confidence interval for the team.
        low=own-r[:,None]; high=own+r[:,None]
        bounds={
            "width_m":[max(0,float(low[:,1].max()-high[:,1].min())),float(high[:,1].max()-low[:,1].min())],
            "depth_m":[max(0,float(low[:,0].max()-high[:,0].min())),float(high[:,0].max()-low[:,0].min())],
            "defensive_line_m":[float(np.sort(low[:,0])[:4].mean()),float(np.sort(high[:,0])[:4].mean())],
        }
    perception_unverified = getattr(m, "meta", {}).get("source", {}).get("type") == "computer_vision"
    return dict(status="estimated" if inferred else "observed",metrics={k:round(v,1) for k,v in point.items()},
                observed_outfield=len(js)-len(inferred),inferred_outfield=len(inferred),
                perception_unverified=perception_unverified,
                sensitivity_bounds_m={k:[round(a,1),round(b,1)] for k,(a,b) in bounds.items()} if bounds else None,
                uncertainty="per-player radius sensitivity; joint coverage unknown" if inferred and known else "uncalibrated" if inferred else "observations only; perception error not included",
                timestamp=float(frame["t"]))


def reason_from_positions(moment,evidence):
    """Use only existing measured deltas; purpose and trade-offs remain hypotheses."""
    deltas=moment.get("deltas",{})
    choices=[("def_line_oop_m","defensive line"),("width_m","team width"),("depth_m","team depth")]
    changes=[(k,label,float(deltas[k])) for k,label in choices if deltas.get(k) is not None and np.isfinite(deltas[k]) and abs(deltas[k])>=2]
    if not changes:
        observation="No supported change of at least 2 m in the available shape measurements."
        interpretation="Position data alone does not establish a tactical change here."
        mechanism="Use the observed event sequence to interpret this moment."
        tradeoff="A positional trade-off cannot be resolved from these measurements."
    else:
        key,label,value=max(changes,key=lambda z:abs(z[2]))
        observation=f"The calculated {label} changed {value:+.1f} m in the comparison windows."
        if key=="def_line_oop_m":
            interpretation="This is consistent with squeezing midfield space." if value>0 else "This is consistent with protecting space behind the defence."
            mechanism="A higher line may reduce the space between teams, provided midfield moves with it." if value>0 else "A deeper line may reduce the room for runs behind, provided the team stays connected."
            tradeoff="More space may become available behind the line." if value>0 else "The opponent may gain territory and room in front of the block."
        elif key=="width_m":
            interpretation="This is consistent with stretching play across the pitch." if value>0 else "This is consistent with concentrating players in central space."
            mechanism="Wider spacing may open central passing lanes if opponents follow the wide players." if value>0 else "Narrower spacing may improve central cover and short passing support."
            tradeoff="Longer distances may reduce support after losing possession." if value>0 else "Wide areas may become easier for opponents to access."
        else:
            interpretation="This is consistent with occupying more vertical space." if value>0 else "This is consistent with bringing the team lines closer together."
            mechanism="Greater depth may create options at different heights, depending on opponent coverage." if value>0 else "Reduced depth may shorten supporting distances, depending on line organisation."
            tradeoff="Larger gaps between lines may become vulnerable." if value>0 else "Less depth may reduce options for playing behind the opposition."
    uncertain=evidence.get("uncertainty")=="uncalibrated" or evidence.get("perception_unverified",False)
    if uncertain:
        interpretation="Tentative: "+interpretation
    outcome=moment.get("outcome",{})
    return dict(has_supported_change=bool(changes),observation=observation,interpretation=interpretation,mechanism=mechanism,
                observed_outcome=outcome.get("description") or outcome.get("type","No recorded outcome"),
                tradeoff=tradeoff,confidence="Position accuracy is unverified; tactical interpretation is tentative." if uncertain else "Tactical intent is inferred; sensitivity bounds are not proof of the mechanism.",
                alternatives="Changes in possession, opposition movement, or perception errors could also explain the positions.",
                retrospective=True,position_evidence=evidence)
