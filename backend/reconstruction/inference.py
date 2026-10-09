"""Inference interface for hidden-player state (decoupled from API / frontend).

    predict_hidden_state(observations, previous_states, ball_state, match_context) -> [player state]

Estimators: "formation" (team-shape, deterministic) or "ml" (learned correction trained on DFL ground truth,
cv/weights/hidden_{mode}_f{fold}.pt). Live mode only ever sees the buffer up to t (causal by construction).
"""
from pathlib import Path

import numpy as np

from backend.reconstruction.datasets import L, W
from backend.reconstruction.estimator import EV, SRC_INF, SRC_OBS, reconstruct

WEIGHTS = Path(__file__).resolve().parents[2] / "cv" / "weights"
_models = {}


# production model: all 7 DFL matches, v3 features, refit (accuracy = the 7-fold estimate); falls back to fold 0
PRODUCTION = "final_v3"
VIDEO_RADIUS_SCALE = 2.67  # broadcast-video domain: fitted on the first half of the SoccerNet clip's reappearances
                           # (second half: 80% radius coverage 0.42 -> 0.92); see cv/clip_eval.py


def default_fold():
    return PRODUCTION if all((WEIGHTS / f"hidden_{m}_{PRODUCTION}.pt").exists() for m in ("live", "replay")) else 0


def load_ml(mode, fold=None):
    fold = default_fold() if fold is None else fold
    key = (mode, fold)
    if key not in _models:
        import torch
        from backend.reconstruction.ml import make_model
        name = fold if isinstance(fold, str) else f"f{fold}"  # e.g. 0 -> f0, "lomo3" (leave-one-match-out fold)
        ck = torch.load(WEIGHTS / f"hidden_{mode}_{name}.pt", weights_only=False)
        net = make_model(ck["n_in"])
        net.load_state_dict(ck["state"])
        net.eval()
        _models[key] = (net, ck["calibration"], ck.get("feat_version", 1))
    return _models[key]


def ml_available(fold=None):
    fold = default_fold() if fold is None else fold
    name = fold if isinstance(fold, str) else f"f{fold}"
    return all((WEIGHTS / f"hidden_{m}_{name}.pt").exists() for m in ("live", "replay"))


def reconstruct_ml(m, vis, mode, on, role, anchors, teams, roles_coarse, alive=None, fold=None, radius_scale=1.0):
    """Formation prior + learned correction + empirically calibrated uncertainty."""
    from backend.reconstruction.ml import apply_calibration, features, predict
    rec = reconstruct(m, vis, mode, {}, on=on, role=role, anchors=anchors)
    net, cal, fv = load_ml(mode, fold)
    X, idx = features(np.where(vis[..., None], m.xy, np.nan), vis, on, teams, roles_coarse, m.ball, m.poss,
                      alive if alive is not None else np.ones(m.T, bool), rec.pos, mode, version=fv)
    if len(idx):
        corr, sig = predict(net, X)
        t, j = idx[:, 0], idx[:, 1]
        corr = np.where((teams[j] == "B")[:, None], -corr, corr)
        rec.pos[t, j] = np.clip(rec.pos[t, j] + corr, [0, 0], [L, W])
        r80, p5 = apply_calibration(cal, sig)
        rec.rad[t, j], rec.conf[t, j] = r80 * radius_scale, p5
    # physical prior: an unseen goalkeeper stays in his own zone (<= 25 m from his goal line)
    for jj, o in enumerate(m.order):
        if role.get(o) == "GK":
            inf_rows = rec.src[:, jj] == SRC_INF
            if teams[jj] == "A":
                rec.pos[inf_rows, jj, 0] = np.minimum(rec.pos[inf_rows, jj, 0], 25)
            else:
                rec.pos[inf_rows, jj, 0] = np.maximum(rec.pos[inf_rows, jj, 0], L - 25)
    fold = default_fold() if fold is None else fold
    rec.meta["estimator"] = f"ml_{mode}_" + (fold if isinstance(fold, str) else f"f{fold}")
    return rec


def coverage(src_row, conf_row, high=0.7):
    """Visibility / evidence coverage for one second."""
    on = src_row > 0
    obs = int((src_row == SRC_OBS).sum())
    inf = src_row == SRC_INF
    hc = int((inf & (conf_row >= high)).sum())
    n = int(on.sum())
    cov = (obs + float(conf_row[inf].sum())) / max(1, n)
    return {"observed_players": obs, "inferred_players": int(inf.sum()), "high_confidence_inferred": hc,
            "coverage": round(cov, 3)}


def predict_hidden_state(observations, previous_states, ball_state, match_context):
    """Single-step causal API.

    observations    {player_id: (x, y)} visible now
    previous_states list of earlier {"observations": {...}, "ball": (x, y), "possession": "A"|"B"|None} (oldest first)
    ball_state      {"x":, "y":, "possession": "A"|"B"|None}
    match_context   {"players": {pid: {"team": "A"|"B", "role": "GK"|..., "anchor": (dx, dy)}}, "estimator": "formation"|"ml"}
    Returns [{"player_id","position","state_source","confidence","uncertainty_radius","evidence"}] for time t.
    """
    from types import SimpleNamespace
    players = match_context["players"]
    order = [f"{p['team']}:{pid}" for pid, p in players.items()]
    pids = list(players)
    steps = list(previous_states) + [{"observations": observations, "ball": (ball_state["x"], ball_state["y"]),
                                      "possession": ball_state.get("possession")}]
    T, N = len(steps), len(pids)
    xy = np.full((T, N, 2), np.nan)
    vis = np.zeros((T, N), bool)
    for t, st in enumerate(steps):
        for j, pid in enumerate(pids):
            if pid in st["observations"]:
                xy[t, j] = st["observations"][pid]
                vis[t, j] = True
    ball = np.array([s["ball"] for s in steps], dtype=float)
    poss = np.array([{"A": 0, "B": 1}.get(s.get("possession"), -1) for s in steps])
    m = SimpleNamespace(xy=xy, ball=ball, poss=poss, order=order, T=T, idx={o: j for j, o in enumerate(order)}, players=[], events=[])
    on = np.ones((T, N), bool)
    role = {o: players[pid].get("role", "CM") for o, pid in zip(order, pids)}
    anchors = {o: players[pid]["anchor"] for o, pid in zip(order, pids)} if all("anchor" in p for p in players.values()) else None
    if match_context.get("estimator") == "ml" and ml_available():
        teams = np.array([players[p]["team"] for p in pids])
        rc = np.array([players[p].get("role_coarse", players[p].get("role", "CM")) for p in pids])
        rec = reconstruct_ml(m, vis, "live", on, role, anchors, teams, rc)
    else:
        rec = reconstruct(m, vis, "live", {}, on=on, role=role, anchors=anchors)
    t = T - 1
    return [{"player_id": pid, "position": {"x": round(float(rec.pos[t, j, 0]), 1), "y": round(float(rec.pos[t, j, 1]), 1)},
             "state_source": "observed" if rec.src[t, j] == SRC_OBS else "inferred",
             "confidence": round(float(rec.conf[t, j]), 2), "uncertainty_radius": round(float(rec.rad[t, j]), 1),
             "evidence": [k for k in EV if rec.evid[t, j] & EV[k]]} for j, pid in enumerate(pids)]
