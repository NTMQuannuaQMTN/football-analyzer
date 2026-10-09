"""Training on SoccerNet-GSR: (1) detector fine-tune, (2) jersey-number classifier.

    .venv/bin/python -m cv.gsr_train export-yolo        # YOLO dataset (train split; valid clips 13+ as val)
    .venv/bin/python -m cv.gsr_train train-yolo --epochs 8
    .venv/bin/python -m cv.gsr_train export-jersey
    .venv/bin/python -m cv.gsr_train train-jersey --epochs 6

Valid clips 1-12 are held out for cv.gsr evaluation (never used for training or model selection).
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from cv.gsr import GSR, OUT, clips, load_clip

CLASSES = {"player": 0, "goalkeeper": 1, "referee": 2, "ball": 3}
YOLO_DIR = OUT / "yolo"
JERSEY_DIR = OUT / "jersey"
WEIGHTS = Path(__file__).resolve().parent / "weights"
HELD_OUT = 12
CROP_W, CROP_H = 48, 96


def split_clips(name):
    if name == "train":
        return clips("train")
    if name == "val":
        return clips("valid")[HELD_OUT:]
    if name == "test":
        return clips("valid")[:HELD_OUT]
    raise ValueError(name)


# ----------------------------------------------------------------------------- detector
def export_yolo(stride=10):
    for part in ("train", "val"):
        img_dir, lab_dir = YOLO_DIR / "images" / part, YOLO_DIR / "labels" / part
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for clip in split_clips(part):
            _, frames = load_clip(clip)
            for f in frames[::stride]:
                stem = f"{clip.name}_{f['file'].stem}"
                dst = img_dir / f"{stem}.jpg"
                if not dst.exists():
                    os.symlink(f["file"], dst)
                rows = []
                for p in f["people"]:
                    if p["role"] not in CLASSES:
                        continue
                    x1, y1, x2, y2 = p["box"]
                    w, h = f["w"], f["h"]
                    rows.append(f"{CLASSES[p['role']]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
                (lab_dir / f"{stem}.txt").write_text("\n".join(rows))
                n += 1
        print(part, n, "images")
    (YOLO_DIR / "data.yaml").write_text(
        f"path: {YOLO_DIR}\ntrain: images/train\nval: images/val\nnames:\n" +
        "".join(f"  {v}: {k}\n" for k, v in CLASSES.items()))


def train_yolo(epochs, imgsz, batch, device, base="yolo11s.pt", fraction=0.5):
    """Laptop-sized default (24 GB unified memory): YOLO11s @1280, half the frames (neighbours are
    near-duplicates). On an Azure ML GPU use base=yolo11m.pt, fraction=1.0, batch=16."""
    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS / base))
    model.train(data=str(YOLO_DIR / "data.yaml"), epochs=epochs, imgsz=imgsz, batch=batch, device=device,
                project=str(OUT / "runs"), name="yolo_gsr", exist_ok=True, workers=2, close_mosaic=1,
                fliplr=0.5, mosaic=0.5, plots=False, fraction=fraction)
    best = OUT / "runs" / "yolo_gsr" / "weights" / "best.pt"
    (WEIGHTS / "yolo_gsr.pt").write_bytes(best.read_bytes())
    print("saved", WEIGHTS / "yolo_gsr.pt")


# ----------------------------------------------------------------------------- jersey numbers
def crop(img, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    h = y2 - y1
    c = img[max(0, y1):max(0, y1 + int(0.6 * h)), max(0, x1):max(0, x2)]  # upper body: where numbers are
    if c.size == 0 or c.shape[0] < 12 or c.shape[1] < 6:
        return None
    return cv2.resize(c, (CROP_W, CROP_H), interpolation=cv2.INTER_AREA)


def export_jersey(stride=15):
    JERSEY_DIR.mkdir(parents=True, exist_ok=True)
    for part in ("train", "val", "test"):
        X, y, track = [], [], []
        for clip in split_clips(part):
            _, frames = load_clip(clip)
            for f in frames[::stride]:
                img = None
                for p in f["people"]:
                    j = p["jersey"]
                    if p["role"] not in ("player", "goalkeeper") or j in (None, "") or not str(j).isdigit() or not 1 <= int(j) <= 99:
                        continue
                    if img is None:
                        img = cv2.imread(str(f["file"]))
                    c = crop(img, p["box"])
                    if c is not None:
                        X.append(c)
                        y.append(int(j))
                        track.append(f"{clip.name}:{p['track']}")
        np.savez_compressed(JERSEY_DIR / f"{part}.npz", X=np.array(X, dtype=np.uint8), y=np.array(y), track=np.array(track))
        print(part, len(X), "crops")


class DigitNet:
    """ResNet18 trunk with digit heads: n_digits (1/2), tens (0-9), units (0-9).
    Sharing digits across numbers forces the model to read, not to recognise players."""

    @staticmethod
    def build():
        import torch
        import torchvision
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        feat = m.fc.in_features
        m.fc = torch.nn.Identity()

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.trunk = m
                self.nd = torch.nn.Linear(feat, 2)
                self.tens = torch.nn.Linear(feat, 10)
                self.units = torch.nn.Linear(feat, 10)

            def forward(self, x):
                h = self.trunk(x)
                return self.nd(h), self.tens(h), self.units(h)
        return Net()


def number_probs(nd, tens, units):
    """Combine heads into P(number) for 1..99 (index = number)."""
    import torch
    pn, pt, pu = torch.softmax(nd, 1), torch.softmax(tens, 1), torch.softmax(units, 1)
    P = torch.zeros(len(pn), 100, device=pn.device)
    P[:, 1:10] = pn[:, :1] * pu[:, 1:10]
    for t in range(1, 10):
        P[:, t * 10:t * 10 + 10] = pn[:, 1:2] * pt[:, t:t + 1] * pu
    return P


def to_tensor(X):
    import torch
    x = torch.from_numpy(X[..., ::-1].copy()).permute(0, 3, 1, 2).float() / 255  # BGR->RGB
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std


def augment(xb, rng):
    """Break the kit-colour / body-shape shortcut: grey, channel swaps, inversion, erasing, jitter."""
    xb = xb.astype(np.float32)
    n = len(xb)
    g = rng.random(n) < 0.5
    xb[g] = xb[g].mean(axis=3, keepdims=True)
    for i in np.where(rng.random(n) < 0.5)[0]:
        xb[i] = xb[i][..., rng.permutation(3)]
    inv = rng.random(n) < 0.2
    xb[inv] = 255 - xb[inv]
    xb = xb * rng.uniform(0.6, 1.4, (n, 1, 1, 1)) + rng.uniform(-30, 30, (n, 1, 1, 1))
    for i in np.where(rng.random(n) < 0.5)[0]:
        h, w = rng.integers(10, 35), rng.integers(8, 22)
        y0, x0 = rng.integers(0, CROP_H - h), rng.integers(0, CROP_W - w)
        xb[i, y0:y0 + h, x0:x0 + w] = rng.uniform(0, 255)
    xb = np.roll(xb, rng.integers(-4, 5), axis=2)
    xb = np.roll(xb, rng.integers(-4, 5), axis=1)
    return np.clip(xb, 0, 255).astype(np.uint8)


def predict_jersey(model, X, device, bs=512):
    import torch
    out = []
    with torch.no_grad():
        for k in range(0, len(X), bs):
            out.append(number_probs(*model(to_tensor(X[k:k + bs]).to(device))).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 100))


def eval_jersey(model, part, device, conf=0.5, min_votes=2):
    d = np.load(JERSEY_DIR / f"{part}.npz")
    P = predict_jersey(model, d["X"], device)
    crop_acc = float(np.mean(P.argmax(1) == d["y"]))
    votes = defaultdict(lambda: np.zeros(100))
    nvotes = defaultdict(int)
    truth = {}
    for p, t, yy in zip(P, d["track"], d["y"]):
        truth[t] = yy
        if p.max() >= conf:  # only legible crops vote
            votes[t] += np.log(p + 1e-6)
            nvotes[t] += 1
    decided = [t for t in truth if nvotes[t] >= min_votes]
    prec = float(np.mean([votes[t].argmax() == truth[t] for t in decided])) if decided else 0.0
    return {"crop_accuracy": round(crop_acc, 3), "track_precision": round(prec, 3),
            "track_coverage": round(len(decided) / max(1, len(truth)), 3), "tracklets": len(truth)}


def train_jersey(epochs, device):
    import torch
    tr = np.load(JERSEY_DIR / "train.npz")
    X, yy = tr["X"], tr["y"]
    nd = torch.from_numpy((yy >= 10).astype(np.int64))
    tens = torch.from_numpy((yy // 10).astype(np.int64))
    units = torch.from_numpy((yy % 10).astype(np.int64))
    model = DigitNet.build().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=epochs * (len(X) // 128 + 1))
    rng = np.random.default_rng(0)
    ce = torch.nn.functional.cross_entropy
    best, best_state = -1, None
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(X))
        tot = 0.0
        for k in range(0, len(X), 128):
            b = perm[k:k + 128]
            o_nd, o_t, o_u = model(to_tensor(augment(X[b], rng)).to(device))
            bt = torch.from_numpy(b)
            two = nd[bt].to(device) == 1
            loss = ce(o_nd, nd[bt].to(device)) + ce(o_u, units[bt].to(device))
            if two.any():
                loss = loss + ce(o_t[two], tens[bt].to(device)[two])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(b)
        model.eval()
        v = eval_jersey(model, "val", device)
        score = v["track_precision"] * v["track_coverage"]
        if score > best:
            best, best_state = score, {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        print(f"epoch {ep + 1}: loss {tot / len(X):.3f} | val {v}", flush=True)
    model.load_state_dict(best_state)
    torch.save(best_state, WEIGHTS / "jersey_digits.pt")
    res = {"val": eval_jersey(model, "val", device), "test (held-out valid clips 1-12)": eval_jersey(model, "test", device)}
    json.dump(res, open(OUT / "jersey_metrics.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export-yolo", "train-yolo", "export-jersey", "train-jersey"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    {"export-yolo": lambda: export_yolo(), "train-yolo": lambda: train_yolo(a.epochs, a.imgsz, a.batch, a.device),
     "export-jersey": lambda: export_jersey(), "train-jersey": lambda: train_jersey(a.epochs, a.device)}[a.cmd]()
