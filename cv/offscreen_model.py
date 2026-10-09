"""Learned off-screen player estimation, trained on Metrica Sports open tracking data.

Broadcast video only shows ~half the players. Metrica sample games contain every player's
position, so we *simulate* a broadcast camera on them (window following the ball, zoom changes,
cuts to close-ups, missed detections) and train an MLP to predict where hidden players are from:
last-seen position, time since seen, ball now / then, visible teammates and opponents, role anchor
in the team shape, and possession.

    .venv/bin/python -m cv.offscreen_model        # train on game 1, evaluate on game 2

Outputs cv/weights/offscreen_mlp.pt and cv/weights/offscreen_metrics.json.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
METRICA = ROOT / "data" / "metrica"
WEIGHTS = Path(__file__).resolve().parent / "weights"
L, W = 105.0, 68.0
N_FEAT = 17


# ----------------------------------------------------------------------------- data
def load_metrica(game: int):
    """1 Hz arrays: ball (T,2) metres, pos {team: (T,n,2)}, period (T,)."""
    out = {}
    for side, team in (("Home", "H"), ("Away", "A")):
        f = METRICA / f"Sample_Game_{game}_RawTrackingData_{side}_Team.csv"
        raw = np.genfromtxt(f, delimiter=",", skip_header=3)
        raw = raw[(raw[:, 1] % 25) == 1]  # 25 fps -> 1 Hz
        out[team] = raw
    a = out["H"]
    period = a[:, 0].astype(int)
    xy = lambda r: np.stack([r[:, 0::2] * L, r[:, 1::2] * W], axis=-1)
    home = xy(out["H"][:, 3:-2])
    away = xy(out["A"][:, 3:-2])
    ball = np.stack([a[:, -2] * L, a[:, -1] * W], axis=1)
    return {"period": period, "ball": ball, "pos": {"H": home, "A": away}}


def attacks_right(P) -> bool:
    """Direction from the goalkeeper (player staying closest to a goal line), not the team mean:
    a dominant team can spend a whole half in the opponent's half."""
    with np.errstate(all="ignore"):
        mx = np.nanmean(P[..., 0], axis=0)
    mx = mx[~np.isnan(mx)]
    gk = mx[np.argmin(np.minimum(mx, L - mx))]
    return bool(gk < L / 2)


def to_own(p, attacks_right: bool):
    return p if attacks_right else np.stack([L - p[..., 0], W - p[..., 1]], axis=-1)


def possession(ball, pos_team, pos_opp):
    """+1 team, -1 opponent, 0 unknown (nearest player within 3 m, carried forward)."""
    T = len(ball)
    out = np.zeros(T)
    cur = 0
    for t in range(T):
        if np.isnan(ball[t, 0]):
            out[t] = cur
            continue
        dt = np.nanmin(np.linalg.norm(pos_team[t] - ball[t], axis=1)) if np.any(~np.isnan(pos_team[t, :, 0])) else 99
        do = np.nanmin(np.linalg.norm(pos_opp[t] - ball[t], axis=1)) if np.any(~np.isnan(pos_opp[t, :, 0])) else 99
        if min(dt, do) < 3:
            cur = 1 if dt < do else -1
        out[t] = cur
    return out


def simulate_camera(ball, positions, rng):
    """Visibility mask (T, N) mimicking a broadcast main camera + cuts + detector misses."""
    T, N = positions.shape[:2]
    vis = np.zeros((T, N), dtype=bool)
    cam = np.nan_to_num(ball[0, 0], nan=L / 2)
    zoom, zoom_left, cut_left = 1.0, 0, 0
    for t in range(T):
        bx = ball[t, 0]
        if not np.isnan(bx):
            cam += 0.5 * (bx - cam)
        cam = float(np.clip(cam, 20, 85))
        if zoom_left <= 0:
            zoom, zoom_left = rng.uniform(0.8, 1.3), rng.integers(8, 25)
        zoom_left -= 1
        if cut_left > 0:
            cut_left -= 1
            continue
        if rng.random() < 0.06:
            cut_left = rng.integers(3, 15)
            continue
        p = positions[t]
        half_w = (17 + 0.22 * (W - p[:, 1])) * zoom
        vis[t] = (np.abs(p[:, 0] - cam) < half_w) & ~np.isnan(p[:, 0]) & (rng.random(N) > 0.12)
    return vis


# ----------------------------------------------------------------------------- features
def centroid_prior(ball_own, poss, coef):
    """Team centroid guess from the ball when nothing is visible (fitted on Metrica)."""
    return np.array([coef[0] * ball_own[0] + coef[1] * poss + coef[2], coef[3] * ball_own[1] + coef[4]])


def features(last, dt, never, ball, ball_last, tm, tm_n, op, op_n, anchor, poss):
    """All positions in the team's own frame (metres). Returns float32 vector of size N_FEAT."""
    return np.array([
        last[0] / L, last[1] / W, min(dt, 60) / 60, float(never),
        ball[0] / L, ball[1] / W, ball_last[0] / L, ball_last[1] / W,
        tm[0] / L, tm[1] / W, tm_n / 10, op[0] / L, op[1] / W, op_n / 10,
        anchor[0] / 30, anchor[1] / 30, poss,
    ], dtype=np.float32)


def build_samples(game: int, seed: int, coef=None):
    d = load_metrica(game)
    rng = np.random.default_rng(seed)
    X, Y, meta = [], [], []
    ball_w = d["ball"]
    for team, opp in (("H", "A"), ("A", "H")):
        P, O = d["pos"][team], d["pos"][opp]
        for per in (1, 2):
            idx = np.where(d["period"] == per)[0]
            if len(idx) == 0:
                continue
            right = attacks_right(P[idx])
            p = to_own(P[idx], right)
            o = to_own(O[idx], right)
            b = to_own(ball_w[idx], right)
            b = np.where(np.isnan(b), np.nanmean(b, axis=0), b)
            poss = possession(b, p, o)
            cen = np.nanmean(p, axis=1)
            anchors = np.nanmean(p - cen[:, None], axis=0)  # role in the shape
            if coef is None:
                ok = ~np.isnan(cen[:, 0])
                A = np.stack([b[ok, 0], poss[ok], np.ones(ok.sum())], 1)
                cx = np.linalg.lstsq(A, cen[ok, 0], rcond=None)[0]
                cy = np.linalg.lstsq(np.stack([b[ok, 1], np.ones(ok.sum())], 1), cen[ok, 1], rcond=None)[0]
                coef = [*cx, *cy]
            vis_p = simulate_camera(b, p, rng)
            vis_o = simulate_camera(b, o, np.random.default_rng(seed + 1))
            vis_o = vis_o | vis_p.any(1, keepdims=True) & (np.abs(o[..., 0] - np.nanmean(np.where(vis_p[..., None], p, np.nan)[..., 0], axis=1, keepdims=True)) < 15)
            n = p.shape[1]
            last = [None] * n
            for t in range(len(idx)):
                tm_vis = p[t][vis_p[t]]
                op_vis = o[t][vis_o[t] & ~np.isnan(o[t, :, 0])]
                tm = tm_vis.mean(0) if len(tm_vis) else b[t]
                op_c = op_vis.mean(0) if len(op_vis) else b[t]
                for i in range(n):
                    if np.isnan(p[t, i, 0]):
                        continue
                    if vis_p[t, i]:
                        last[i] = (p[t, i] + rng.normal(0, 0.6, 2), t, b[t].copy())
                        continue
                    if last[i] is None:
                        guess = centroid_prior(b[t], poss[t], coef) + anchors[i]
                        lp, dt, never, bl = guess, 60, 1, b[t]
                    else:
                        lp, t0, bl = last[i]
                        dt, never = t - t0, 0
                    X.append(features(lp, dt, never, b[t], bl, tm, len(tm_vis), op_c, len(op_vis), anchors[i], poss[t]))
                    Y.append((p[t, i] - lp) / 20)
                    meta.append((dt, never))
    return np.array(X), np.array(Y, dtype=np.float32), np.array(meta), coef


class OffscreenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N_FEAT, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2))

    def forward(self, x):
        return self.net(x)


def mirror(X, Y):
    """Left/right symmetry augmentation."""
    Xm, Ym = X.copy(), Y.copy()
    for j in (1, 5, 7, 9, 12):  # y coordinates normalised by W
        Xm[:, j] = 1 - Xm[:, j]
    Xm[:, 15] = -Xm[:, 15]
    Ym[:, 1] = -Ym[:, 1]
    return np.vstack([X, Xm]), np.vstack([Y, Ym])


def evaluate(model, X, Y, meta):
    with torch.no_grad():
        pred = model(torch.from_numpy(X)).numpy()
    err = {}
    true = Y * 20
    last_ball_shift = np.stack([(X[:, 4] - X[:, 6]) * L, (X[:, 5] - X[:, 7]) * W], 1)
    cands = {"hold_last_seen": np.zeros_like(true), "last_seen_plus_ball_shift": last_ball_shift * np.array([0.45, 0.3]),
             "learned_mlp": pred * 20}
    for name, p in cands.items():
        e = np.linalg.norm(p - true, axis=1)
        buckets = {"all": np.ones(len(e), bool), "hidden<5s": (meta[:, 0] < 5) & (meta[:, 1] == 0),
                   "hidden5-20s": (meta[:, 0] >= 5) & (meta[:, 0] < 20) & (meta[:, 1] == 0),
                   "hidden>20s": (meta[:, 0] >= 20) & (meta[:, 1] == 0), "never_seen": meta[:, 1] == 1}
        err[name] = {k: round(float(e[m].mean()), 2) for k, m in buckets.items() if m.any()}
    return err


def train():
    print("building samples (game 1 = train, game 2 = test)...")
    Xtr, Ytr, _, coef = build_samples(1, seed=0)
    Xte, Yte, Mte, _ = build_samples(2, seed=7, coef=coef)
    Xtr, Ytr = mirror(Xtr, Ytr)
    print(f"train {len(Xtr)} hidden-player samples, test {len(Xte)}")
    torch.manual_seed(0)
    model = OffscreenMLP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xt, Yt = torch.from_numpy(Xtr), torch.from_numpy(Ytr)
    for ep in range(12):
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for k in range(0, len(Xt), 1024):
            b = perm[k:k + 1024]
            loss = nn.functional.smooth_l1_loss(model(Xt[b]), Yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        print(f"epoch {ep + 1}: loss {tot / len(Xt):.4f}")
    model.eval()
    metrics = evaluate(model, Xte, Yte, Mte)
    WEIGHTS.mkdir(exist_ok=True)
    torch.save({"state": model.state_dict(), "centroid_coef": [float(c) for c in coef]}, WEIGHTS / "offscreen_mlp.pt")
    json.dump({"test_game": 2, "mean_error_m": metrics}, open(WEIGHTS / "offscreen_metrics.json", "w"), indent=1)
    print(json.dumps(metrics, indent=1))


def load_model():
    ck = torch.load(WEIGHTS / "offscreen_mlp.pt", weights_only=False)
    m = OffscreenMLP()
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck["centroid_coef"]


if __name__ == "__main__":
    train()
