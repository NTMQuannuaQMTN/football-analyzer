"""Isolated match-held-out reconstruction experiments for the local training session.

Validation alone selects candidates; test match is never used for training,
epoch selection, or uncertainty calibration. Checkpoints stay in --out.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from backend.reconstruction import ml
from backend.reconstruction.real_benchmark import split


def train(tr, va, epochs, seed, deadline):
    torch.manual_seed(seed)
    net = ml.make_model(tr["X"].shape[1])
    # make_model seeds its initial weights; restore the requested shuffle seed.
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=0.0006, weight_decay=0.0002)
    x, y = torch.from_numpy(tr["X"]), torch.from_numpy(tr["Y"])
    xv, yv = torch.from_numpy(va["X"]), torch.from_numpy(va["Y"])
    best, state, history = float("inf"), None, []
    for ep in range(epochs):
        if time.time() >= deadline:
            break
        net.train()
        perm = torch.randperm(len(x))
        for start in range(0, len(x), 2048):
            b = perm[start:start+2048]
            mu, ls = net(x[b])
            ls = ls.clamp(-4, 4)
            d2 = ((mu.detach()-y[b])**2).sum(1)
            loss = torch.nn.functional.huber_loss(mu, y[b], delta=0.25) + .01*(2*ls+d2/(2*torch.exp(2*ls))).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
            opt.step()
        net.eval()
        with torch.no_grad():
            total = 0.0
            for start in range(0, len(xv), 4096):
                mu, _ = net(xv[start:start+4096])
                total += torch.linalg.norm((mu-yv[start:start+4096])*20, dim=1).sum().item()
            error = total / len(xv)
        history.append(error)
        print(f"epoch {ep+1}: validation {error:.3f} m", flush=True)
        if error < best:
            best = error
            state = {k: v.detach().clone() for k,v in net.state_dict().items()}
        if len(history) > 15 and np.argmin(history) < len(history)-12:
            break
    if state is None:
        raise RuntimeError("Budget expired before any training epoch completed")
    net.load_state_dict(state)
    return net, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cv/out/training_20261011/reconstruction")
    ap.add_argument("--deadline", type=float, required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    tr_matches, val, test = split(0)
    records, candidates = [], {}
    for version in (3, 4):
        ml.FEAT_VERSION = version
        tr = ml.build_per_match(tr_matches, ml.TRAIN_MASKS, 2, workers=2, kind="train")
        va = ml.build_per_match([val], ["sc_camera", "k=14", "k=10", "k=6"], 3,
                                seed=55, workers=2, kind="session_val")
        for mode in ("live", "replay"):
            for seed in (11026, 22026):
                if time.time() >= a.deadline:
                    break
                print(f"v{version} {mode} seed {seed}, train {len(tr[mode]['X'])}, val {len(va[mode]['X'])}", flush=True)
                net, hist = train(tr[mode], va[mode], 80, seed, a.deadline)
                corr, sigma = ml.predict(net, va[mode]["X"])
                cal = ml.calibrate(sigma, np.linalg.norm(corr-va[mode]["Y"]*20, axis=1))
                name = f"hidden_{mode}_session_v{version}_s{seed}.pt"
                torch.save({"state": net.state_dict(), "n_in": va[mode]["X"].shape[1],
                            "feat_version": version, "mode": mode, "calibration": cal,
                            "train": tr_matches, "val": val, "test": test, "seed": seed}, out/name)
                row = dict(version=version, mode=mode, seed=seed, checkpoint=name,
                           validation_mae_m=min(hist), history=hist,
                           train_samples=len(tr[mode]["X"]), validation_samples=len(va[mode]["X"]))
                records.append(row)
                if mode not in candidates or min(hist)<candidates[mode][0]:
                    candidates[mode] = (min(hist), net, cal, version, name)
                (out/"progress.json").write_text(json.dumps(dict(train=tr_matches,val=val,test=test,experiments=records),indent=2))
        del tr, va
    # Selected live/replay may have different schemas. Evaluate separately so
    # the existing evaluator receives the correct global feature version.
    tests = {}
    for mode, (_, net, cal, version, name) in candidates.items():
        ml.FEAT_VERSION = version
        # Evaluator expects both modes; evaluate one selected model at a time
        # via the dedicated evaluator rather than passing incompatible nets.
        from backend.reconstruction.session_evaluation import evaluate
        tests[mode] = evaluate(net, cal, mode, version, test)
        tests[mode]["checkpoint"] = name
    (out/"results.json").write_text(json.dumps(dict(train=tr_matches,val=val,test=test,experiments=records,selected_test=tests),indent=2))


if __name__ == "__main__":
    main()
