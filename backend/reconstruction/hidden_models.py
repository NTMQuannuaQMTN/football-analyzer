"""Hidden-player models shared by training, benchmark and app inference (one preprocessing path).

Every model predicts a CORRECTION to the team-shape prior (`estimator.reconstruct`) for each hidden on-pitch
player-second, in the player's own-team frame (attacking +x), plus a log sigma that is calibrated empirically.

  mlp       the existing v3 row model (`ml.make_model`): structured features of the current second only
  gru_ctx   v3 row features + a GRU over a window of per-second context for that player (own track when visible,
            team-shape prior, ball, visible team / opponent centroids and counts, possession, ball status, cut).
            live: window t-W+1 .. t (past only). replay: window t-R .. t+R (labelled retrospective).
  ensemble  mean of several members (any arch); sigma = sqrt(mean sigma^2 + spread of the means), recalibrated

Checkpoints are dicts with "arch"; legacy checkpoints without it are the v3 MLP.
"""
from pathlib import Path

import numpy as np

from backend.reconstruction.datasets import L, W

CTX_DIM = 18
LIVE_WINDOW = 20          # seconds of past context (t-19 .. t)
REPLAY_RADIUS = 12        # seconds either side (t-12 .. t+12)


def window_offsets(mode):
    return np.arange(-LIVE_WINDOW + 1, 1) if mode == "live" else np.arange(-REPLAY_RADIUS, REPLAY_RADIUS + 1)


def coarse_roles(roles, role_agnostic):
    roles = np.asarray(roles)
    return np.where(roles == "GK", "GK", "CM") if role_agnostic else roles


def context_tensor(pos_obs, vis, on, team, ball, poss, alive, prior):
    """(T, N, CTX_DIM) float32 per-second context, each player in his own-team frame. Uses only second t's data
    at index t; which seconds a model may read is decided by `window_offsets` (live never reads > t)."""
    T, N = vis.shape
    team = np.asarray(team)
    is_b = team == "B"

    def own(p):  # p (T, N, 2) world -> own frame of each column's team
        q = p.copy()
        q[..., 0] = np.where(is_b[None], L - p[..., 0], p[..., 0])
        q[..., 1] = np.where(is_b[None], W - p[..., 1], p[..., 1])
        return q

    C = np.zeros((T, N, CTX_DIM), np.float32)
    v = vis & on
    obs = own(np.where(v[..., None], np.nan_to_num(pos_obs), 0.0))
    C[..., 0] = v
    C[..., 1] = np.where(v, obs[..., 0] / L, 0)
    C[..., 2] = np.where(v, obs[..., 1] / W, 0)
    pr = own(np.nan_to_num(prior, nan=0.0))
    C[..., 3] = np.where(on, pr[..., 0] / L, 0)
    C[..., 4] = np.where(on, pr[..., 1] / W, 0)
    bok = np.isfinite(ball).all(1)
    bw = np.where(bok[:, None], ball, [L / 2, W / 2])
    bx = np.where(is_b[None], L - bw[:, 0:1], bw[:, 0:1])
    by = np.where(is_b[None], W - bw[:, 1:2], bw[:, 1:2])
    C[..., 5], C[..., 6], C[..., 7] = bx / L, by / W, bok[:, None]
    cen, cnt = {}, {}
    for tm in ("A", "B"):
        m = v & (team == tm)[None]
        cnt[tm] = m.sum(1)
        s = np.where(m[..., None], np.nan_to_num(pos_obs), 0).sum(1)
        cen[tm] = np.where(cnt[tm][:, None] > 0, s / np.maximum(cnt[tm], 1)[:, None], bw)
    for j in range(N):
        mine, other = ("B", "A") if is_b[j] else ("A", "B")
        for k, tm in ((8, mine), (11, other)):
            c = cen[tm] if not is_b[j] else np.stack([L - cen[tm][:, 0], W - cen[tm][:, 1]], 1)
            C[:, j, k] = cnt[tm] / 11
            C[:, j, k + 1] = c[:, 0] / L
            C[:, j, k + 2] = c[:, 1] / W
        mine_i = 1 if is_b[j] else 0
        C[:, j, 14] = np.where(poss < 0, 0.0, np.where(poss == mine_i, 1.0, -1.0))
    C[..., 15] = (alive if alive is not None else np.ones(T, bool))[:, None]
    C[..., 16] = (v.sum(1) == 0)[:, None]
    C[..., 17] = 1.0  # in-range flag (padding rows are all zeros)
    return C


class CtxStore:
    """Concatenated, zero-padded context tensors so a batch of windows is one fancy index."""

    def __init__(self, tensors, mode):
        self.off = window_offsets(mode)
        pad = int(np.abs(self.off).max()) + 1
        n = max(c.shape[1] for c in tensors)
        bases, blocks, b = [], [], 0
        for c in tensors:
            blk = np.zeros((len(c) + 2 * pad, n, CTX_DIM), np.float16)
            blk[pad:pad + len(c), :c.shape[1]] = c
            blocks.append(blk)
            bases.append(b + pad)
            b += len(blk)
        self.data = np.concatenate(blocks) if blocks else np.zeros((0, n, CTX_DIM), np.float16)
        self.bases = np.array(bases, dtype=np.int64)

    def windows(self, inst, t, j):
        rows = (self.bases[inst] + t)[:, None] + self.off[None, :]
        return self.data[rows, j[:, None]].astype(np.float32)


def make_gru(n_in, mode, hidden=64):
    import torch
    from torch import nn

    class GRUCtx(nn.Module):
        def __init__(self):
            super().__init__()
            self.bi = mode == "replay"
            self.gru = nn.GRU(CTX_DIM, hidden, batch_first=True, bidirectional=self.bi)
            h = hidden * (2 if self.bi else 1)
            self.body = nn.Sequential(nn.Linear(n_in + h, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                                      nn.Linear(256, 128), nn.ReLU())
            self.mu = nn.Linear(128, 2)
            self.logsig = nn.Linear(128, 1)

        def forward(self, x, seq):
            _, hN = self.gru(seq)
            h = torch.cat([hN[0], hN[1]], 1) if self.bi else hN[0]
            z = self.body(torch.cat([x, h], 1))
            return self.mu(z), self.logsig(z).squeeze(-1)
    torch.manual_seed(0)
    return GRUCtx()


# ----------------------------------------------------------------------------- loading / prediction
class HiddenModel:
    """Uniform predict(): corr (n, 2) metres in own frame, sigma (n,) raw scale, then calibrated r80 / p5."""

    def __init__(self, ck, members=None):
        self.ck = ck
        self.arch = ck.get("arch", "mlp")
        self.mode = ck.get("mode")
        self.feat_version = ck.get("feat_version", 3)
        self.role_agnostic = bool(ck.get("role_agnostic", False))
        self.calibration = ck["calibration"]
        self.members = members
        self.net = None
        if self.arch == "mlp":
            from backend.reconstruction.ml import make_model
            self.net = make_model(ck["n_in"])
        elif self.arch == "gru_ctx":
            self.net = make_gru(ck["n_in"], ck["mode"], ck.get("hidden", 64))
        if self.net is not None:
            self.net.load_state_dict(ck["state"])
            self.net.eval()

    def calibrate(self, sig, X):
        """(r80, p5) per row. With a horizon table (calibration_h, fitted on the validation tail) the empirical radius and
        P(err <= 5 m) come from sigma deciles WITHIN the row's time-unseen bucket; otherwise the global sigma deciles."""
        from backend.reconstruction.ml import apply_calibration
        tabs = self.ck.get("calibration_h")
        if not tabs or not len(sig):
            return apply_calibration(self.calibration, sig)
        b = horizon_bucket(X)
        r80, p5 = apply_calibration(self.calibration, sig)
        for name, tab in tabs.items():
            sel = b == name
            if sel.any():
                r80[sel], p5[sel] = apply_calibration(tab, sig[sel])
        return r80, p5

    @property
    def needs_context(self):
        if self.arch == "ensemble":
            return any(m.needs_context for m in self.members)
        return self.arch == "gru_ctx"

    def raw(self, X, seq=None):
        import torch
        if self.arch == "ensemble":
            outs = [m.raw(X, seq) for m in self.members]
            mus = np.stack([o[0] for o in outs])
            sig2 = np.mean([o[1] ** 2 for o in outs], 0) + (mus.var(0).sum(1) / 400)
            return mus.mean(0), np.sqrt(sig2)
        mus, sigs = [], []
        with torch.no_grad():
            for k in range(0, len(X), 8192):
                x = torch.from_numpy(X[k:k + 8192])
                if self.arch == "gru_ctx":
                    mu, ls = self.net(x, torch.from_numpy(seq[k:k + 8192]))
                else:
                    mu, ls = self.net(x)
                mus.append(mu.numpy())
                sigs.append(np.exp(ls.numpy()))
        if not mus:
            return np.zeros((0, 2)), np.zeros(0)
        return np.concatenate(mus) * 20, np.concatenate(sigs)


HORIZON_BUCKETS = (("0-10s", 0, 10), ("11-30s", 11, 30), ("31-120s", 31, 120), ("120s+", 121, 10 ** 9))


def horizon_bucket(X):
    """Time-unseen bucket per feature row (v3 rows: col 4 = log1p(min(since, 600)) / 4, col 5 = never seen)."""
    since = np.expm1(np.asarray(X)[:, 4] * 4)
    never = np.asarray(X)[:, 5] > 0.5
    out = np.full(len(since), "never", dtype=object)
    for name, a, b in HORIZON_BUCKETS:
        out[(~never) & (since >= a - 0.5) & (since <= b + 0.5)] = name
    return out


def load_hidden(path):
    import torch
    ck = torch.load(Path(path), weights_only=False)
    if ck.get("arch") == "ensemble":
        members = [load_hidden(Path(path).parent / p) for p in ck["members"]]
        return HiddenModel(ck, members)
    return HiddenModel(ck)


def predict_segment(model, pos, vis, on, team, role, ball, poss, alive, prior, mode, t_index=None, return_X=False):
    """Features exactly as in training, then the model. Returns (idx (n,2) [t,j], corr_world (n,2), sigma (n,))."""
    from backend.reconstruction.ml import features
    roles = coarse_roles(role, model.role_agnostic)
    pos_obs = np.where(vis[..., None], pos, np.nan)
    X, idx = features(pos_obs, vis, on, team, roles, ball, poss, alive, prior, mode, t_index=t_index,
                      version=model.feat_version)
    if not len(idx):
        return (idx, np.zeros((0, 2)), np.zeros(0), X) if return_X else (idx, np.zeros((0, 2)), np.zeros(0))
    seq = None
    if model.needs_context:
        C = context_tensor(pos_obs, vis, on, team, ball, poss, alive, prior)
        store = CtxStore([C], mode)
        seq = store.windows(np.zeros(len(idx), np.int64), idx[:, 0], idx[:, 1])
    corr, sig = model.raw(X, seq)
    corr = np.where((np.asarray(team)[idx[:, 1]] == "B")[:, None], -corr, corr)
    if return_X:
        return idx, corr, sig, X
    return idx, corr, sig
