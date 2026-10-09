"""End-to-end check of role-based identity + off-screen estimation on Metrica game 2 (unseen).

Simulates what the video pipeline produces: a broadcast camera view, anonymous track IDs that
restart after every camera cut, team known from shirt colour. Runs RoleEstimator (forward in
time), then the offline smoother that also uses the *next* sighting (smooth_gaps), and compares
both with the true positions of all players.

Two identity modes:
  estimated - identities from formation geometry only (today's pipeline)
  oracle    - every visible player's identity is known (upper bound for jersey-number reading)
"""
import json

import numpy as np
from scipy.optimize import linear_sum_assignment

from cv.offscreen_model import WEIGHTS, attacks_right, load_metrica, possession, simulate_camera, to_own
from cv.roles import RoleEstimator, smooth_gaps


def run_mode(d, mode, seed=3):
    rng = np.random.default_rng(seed)
    acc = {k: [] for k in ("hid_fwd", "hid_smooth", "hid_hold", "set_fwd", "set_smooth", "jump_fwd", "jump_smooth", "id")}
    for per in (1, 2):
        idx = np.where(d["period"] == per)[0]
        H, A = d["pos"]["H"][idx], d["pos"]["A"][idx]
        ball = d["ball"][idx]
        ball = np.where(np.isnan(ball), np.nanmean(ball, axis=0), ball)
        right_h = attacks_right(H)
        teams = {"A": H if right_h else A, "B": A if right_h else H}
        vis = {t: simulate_camera(ball, teams[t], np.random.default_rng(seed + k)) for k, t in enumerate("AB")}
        cut = ~(vis["A"].any(1) | vis["B"].any(1))
        for team in "AB":
            P = teams[team]
            starters = [i for i in range(P.shape[1]) if not np.isnan(P[0, i, 0])][:11]
            own = to_own(P[:, starters], team == "A")
            anchors = np.nanmean(own - np.nanmean(own, axis=1)[:, None], axis=0)
            gk = int(np.nanargmin(anchors[:, 0]))
            roles = [{"role": "GK" if i == gk else f"R{i}", "anchor": tuple(anchors[i]), "line": 0} for i in range(len(starters))]
            est = RoleEstimator(team, roles)
            opp = teams["B" if team == "A" else "A"]
            poss = possession(to_own(ball, team == "A"), own, to_own(opp, team == "A"))
            Tn, n = len(idx), len(starters)
            fwd = np.zeros((Tn, n, 2))
            obs = np.zeros((Tn, n), bool)
            slot_of_player, next_slot = {}, 0
            for t in range(Tn):
                if cut[t] and mode == "estimated":
                    slot_of_player = {}  # tracker IDs don't survive camera cuts
                dets, truth_of_slot = [], {}
                for k, i in enumerate(starters):
                    if vis[team][t, i] and not np.isnan(P[t, i, 0]):
                        if mode == "oracle":
                            sid = k
                        else:
                            if i not in slot_of_player or rng.random() < 0.02:
                                slot_of_player[i], next_slot = next_slot, next_slot + 1
                            sid = slot_of_player[i]
                        dets.append((sid, P[t, i] + rng.normal(0, 0.6, 2), k if mode == "oracle" else None))
                        truth_of_slot[sid] = k
                opp_dets = [opp[t, j] for j in range(opp.shape[1]) if vis["B" if team == "A" else "A"][t, j] and not np.isnan(opp[t, j, 0])]
                world, o = est.step(t, ball[t], poss[t], dets, opp_dets)
                fwd[t], obs[t] = world, o
                for k in range(n):
                    if o[k] and est.slot_of[k] in truth_of_slot:
                        acc["id"].append(truth_of_slot[est.slot_of[k]] == k)
            smooth = np.stack([smooth_gaps(fwd[:, k], obs[:, k]) for k in range(n)], axis=1)
            for k, i in enumerate(starters):
                last = None
                for t in range(Tn):
                    if np.isnan(P[t, i, 0]):
                        continue
                    if vis[team][t, i]:
                        last = P[t, i]
                        continue
                    acc["hid_fwd"].append(np.linalg.norm(fwd[t, k] - P[t, i]))
                    acc["hid_smooth"].append(np.linalg.norm(smooth[t, k] - P[t, i]))
                    if last is not None:
                        acc["hid_hold"].append(np.linalg.norm(last - P[t, i]))
                # "teleports": per-second jumps above 11 m/s in the estimated path
                acc["jump_fwd"].append(np.sum(np.linalg.norm(np.diff(fwd[:, k], axis=0), axis=1) > 11))
                acc["jump_smooth"].append(np.sum(np.linalg.norm(np.diff(smooth[:, k], axis=0), axis=1) > 11))
            for t in range(Tn):
                live = [(k, i) for k, i in enumerate(starters) if not np.isnan(P[t, i, 0]) and not vis[team][t, i]]
                if not live:
                    continue
                T_ = np.array([P[t, i] for _, i in live])
                for key, est_pos in (("set_fwd", fwd[t]), ("set_smooth", smooth[t])):
                    C = np.linalg.norm(T_[:, None] - est_pos[None], axis=2)
                    r, c = linear_sum_assignment(C)
                    acc[key] += [float(C[a, b]) for a, b in zip(r, c)]
    m = lambda k: round(float(np.mean(acc[k])), 2)
    return {
        "off_screen_error_per_player_m": {"frozen_at_last_seen": m("hid_hold"), "forward_model": m("hid_fwd"),
                                          "forward+next_sighting": m("hid_smooth")},
        "off_screen_team_shape_error_m": {"forward_model": m("set_fwd"), "forward+next_sighting": m("set_smooth")},
        "teleports_per_player_half (>11 m/s)": {"forward_model": m("jump_fwd"), "forward+next_sighting": m("jump_smooth")},
        "identity_accuracy_visible": round(float(np.mean(acc["id"])), 3),
    }


def run(game=2):
    d = load_metrica(game)
    out = {mode: run_mode(d, mode) for mode in ("estimated", "oracle")}
    json.dump(out, open(WEIGHTS / "role_eval_metrics.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    run()
