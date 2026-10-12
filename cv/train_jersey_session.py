"""Match-independent, bounded jersey experiment using existing local crops."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cv import gsr_train as g
from cv.gsr import clips


def repartition(root):
    games={p.name:json.loads((p/"Labels-GameState.json").read_text())["info"]["game_id"]
           for split in ("train","valid") for p in clips(split)}
    source=g.JERSEY_DIR
    pieces={k:[] for k in ("train","val","test")}
    for filename in ("train.npz","val.npz","test.npz"):
        with np.load(source/filename) as d:
            track=d["track"]; X=d["X"]; y=d["y"]
            game=np.array([games[t.split(":")[0]] for t in track])
            for part,ids in (("train",["4","6","9"]),("val",["3","5"]),("test",["2"])):
                sel=np.isin(game,ids)
                if sel.any():
                    pieces[part].append((X[sel],y[sel],track[sel]))
    root.mkdir(parents=True,exist_ok=True)
    for part,ps in pieces.items():
        X,y,track=(np.concatenate([p[i] for p in ps]) for i in range(3))
        np.savez_compressed(root/f"{part}.npz",X=X,y=y,track=track)
        print(part,len(X),len(np.unique(track)),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--hours",type=float,default=1)
    ap.add_argument("--out",default="cv/out/training_20261011/jersey")
    ap.add_argument("--prepare-only",action="store_true")
    ap.add_argument("--device",default="mps")
    a=ap.parse_args(); root=Path(a.out); data=root/"data"
    if not (data/"test.npz").exists():
        repartition(data)
    if a.prepare_only:
        return
    # Require local pretrained trunk. Do not let torchvision fetch weights.
    cached=Path(torch.hub.get_dir())/"checkpoints"/"resnet18-f37072fd.pth"
    if not cached.exists():
        raise FileNotFoundError(f"Local ImageNet trunk missing: {cached}")
    g.JERSEY_DIR=data
    torch.manual_seed(11026); torch.set_num_threads(2)
    with np.load(data/"train.npz") as tr:
        X,yy=tr["X"],tr["y"]
    nd=torch.from_numpy((yy>=10).astype(np.int64))
    tens=torch.from_numpy((yy//10).astype(np.int64)); units=torch.from_numpy((yy%10).astype(np.int64))
    model=g.DigitNet.build().to(a.device)
    opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0005)
    rng=np.random.default_rng(11026); ce=torch.nn.functional.cross_entropy
    deadline=time.monotonic()+a.hours*3600
    history=[]; best=-1
    for ep in range(120):
        if time.monotonic()>=deadline:
            break
        model.train(); perm=rng.permutation(len(X)); losses=[]
        for start in range(0,len(X),128):
            if time.monotonic()>=deadline:
                break
            b=perm[start:start+128]; bt=torch.from_numpy(b)
            n,t,u=model(g.to_tensor(g.augment(X[b],rng)).to(a.device))
            two=nd[bt].to(a.device)==1
            loss=ce(n,nd[bt].to(a.device))+ce(u,units[bt].to(a.device))
            if two.any():
                loss+=ce(t[two],tens[bt].to(a.device)[two])
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        model.eval(); metrics=g.eval_jersey(model,"val",a.device)
        score=metrics["track_precision"]*metrics["track_coverage"]
        history.append(dict(epoch=ep+1,loss=float(np.mean(losses)),**metrics))
        print(history[-1],flush=True)
        if score>best:
            best=score
            torch.save({k:t.detach().cpu().clone() for k,t in model.state_dict().items()},root/"best.pt")
        (root/"history.json").write_text(json.dumps(history,indent=2))
        if len(history)>20 and np.argmax([h["track_precision"]*h["track_coverage"] for h in history])<len(history)-12:
            break
    model.load_state_dict(torch.load(root/"best.pt",weights_only=False)); model.eval()
    thresholds=[.5,.7,.85,.95]
    validation={str(c):g.eval_jersey(model,"val",a.device,conf=c) for c in thresholds}
    eligible=[c for c in thresholds if validation[str(c)]["track_precision"]>=.9]
    threshold=max(eligible,key=lambda c:validation[str(c)]["track_coverage"]) if eligible else .95
    test=g.eval_jersey(model,"test",a.device,conf=threshold)
    report=dict(train_games=["4","6","9"],val_games=["3","5"],test_games=["2"],
                validation=validation,selected_threshold=threshold,test=test,
                note="No legibility labels: crop targets include front views without readable numbers; aggregate track voting is retrospective")
    (root/"results.json").write_text(json.dumps(report,indent=2)); print(report,flush=True)


if __name__=="__main__":
    main()
