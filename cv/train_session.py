"""Bounded, isolated local detector experiment; never replaces production weights.

Run with ``python -m cv.train_session --hours 1.6``. Entire GSR game 2 is
held out; games 3/5 select epochs; games 4/6/9 train. No footage downloads.
"""
import argparse
import json
import os
from pathlib import Path

from cv.gsr import clips, load_clip
from cv.gsr_train import CLASSES, WEIGHTS


def export_dataset(out, stride=15):
    out = Path(out).resolve()
    groups = {"train": {"4", "6", "9"}, "val": {"3", "5"}, "test": {"2"}}
    manifest = {"stride": stride, "games": {k: sorted(v) for k, v in groups.items()}, "clips": {}, "frames": {}}
    for part, games in groups.items():
        names, count = [], 0
        for split in ("train", "valid"):
            for clip in clips(split):
                info, frames = load_clip(clip)
                if info["game_id"] not in games:
                    continue
                names.append(clip.name)
                for f in frames[::stride]:
                    stem = f"{clip.name}_{f['file'].stem}"
                    im = out / "images" / part / f"{stem}.jpg"
                    lab = out / "labels" / part / f"{stem}.txt"
                    im.parent.mkdir(parents=True, exist_ok=True)
                    lab.parent.mkdir(parents=True, exist_ok=True)
                    if not im.exists():
                        os.symlink(f["file"].resolve(), im)
                    rows = []
                    for p in f["people"]:
                        if p["role"] not in CLASSES:
                            continue
                        x1, y1, x2, y2 = p["box"]
                        w, h = f["w"], f["h"]
                        rows.append(f"{CLASSES[p['role']]} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
                    lab.write_text("\n".join(rows))
                    count += 1
        manifest["clips"][part], manifest["frames"][part] = names, count
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "data.yaml").write_text(f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n" + "".join(f"  {v}: {k}\n" for k, v in CLASSES.items()))
    return out / "data.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=1.6)
    ap.add_argument("--out", default="cv/out/training_20261011")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--initial", help="optional local validation-selected checkpoint for continuation")
    ap.add_argument("--data", help="reuse an existing isolated dataset manifest")
    ap.add_argument("--optimizer", default="auto", choices=["auto", "AdamW", "SGD"])
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--warmup-epochs", type=float, default=3.0)
    a = ap.parse_args()
    if not 0 < a.hours <= 5:
        ap.error("hours must be between 0 and 5")
    root = Path(a.out).resolve()
    root.mkdir(parents=True,exist_ok=True)
    data = Path(a.data).resolve() if a.data else export_dataset(root / "detector_data")
    from ultralytics import YOLO
    # Fresh generic checkpoint: the previous football model selected epochs
    # on clips from held-out game 2, so it cannot be a clean initialisation.
    model = YOLO(a.initial or str(WEIGHTS / "yolo11s.pt"))
    model.train(data=str(data), epochs=300, time=a.hours, imgsz=960, batch=4,
                device=a.device, project=str(root), name="detector", exist_ok=False,
                workers=2, seed=11026, deterministic=True, close_mosaic=0,
                mosaic=0.3, fliplr=0.5, plots=False, amp=False, save_period=5,
                optimizer=a.optimizer, lr0=a.lr0, warmup_epochs=a.warmup_epochs)
    best = root / "detector" / "weights" / "best.pt"
    if best.exists():
        metrics = YOLO(str(best)).val(data=str(data), split="test", imgsz=960,
                                      device=a.device, batch=4, plots=False)
        (root / "detector_test.json").write_text(json.dumps(metrics.results_dict, indent=2))


if __name__ == "__main__":
    main()
