"""Bounded training and independent video-to-pitch evaluation of named markings."""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from cv.gsr import clips, load_clip, project
from cv.pitch_lines import NAMES, PitchLineNet, heatmap_points, homography_from_lines


def prepare(root, stride=30):
    root.mkdir(parents=True,exist_ok=True)
    groups={"train":{"4","6","9"},"val":{"3","5"},"test":{"2"}}
    rows={k:[] for k in groups}
    for split in ("train","valid"):
        for clip in clips(split):
            d=json.loads((clip/"Labels-GameState.json").read_text())
            part=next(k for k,v in groups.items() if d["info"]["game_id"] in v)
            lines={a["image_id"]:a["lines"] for a in d["annotations"] if a["supercategory"]=="pitch"}
            people={f["file"].name:f["people"] for f in load_clip(clip)[1][::stride]}
            for im in d["images"][::stride]:
                marks=lines.get(im["image_id"],{})
                if not marks:
                    continue
                path=root/f"{clip.name}_{im['file_name']}"
                if not path.exists():
                    image=cv2.imread(str(clip/"img1"/im["file_name"]))
                    if image is None:
                        raise FileNotFoundError(clip/"img1"/im["file_name"])
                    cv2.imwrite(str(path),cv2.resize(image,(512,288)))
                rows[part].append(dict(image=str(path.resolve()),clip=clip.name,
                                       lines=marks,people=people.get(im["file_name"],[])))
    (root/"rows.json").write_text(json.dumps(rows))
    return rows


class LineDataset(Dataset):
    def __init__(self,rows):
        self.rows=rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,i):
        row=self.rows[i]
        image=cv2.cvtColor(cv2.imread(row["image"]),cv2.COLOR_BGR2RGB)
        target=np.zeros((len(NAMES),72,128),np.float32)
        for j,name in enumerate(NAMES):
            pts=row["lines"].get(name,[])
            if len(pts)<2:
                continue
            xy=np.asarray([[p["x"]*127,p["y"]*71] for p in pts],np.int32)
            cv2.polylines(target[j],[xy],False,1,2)
        return torch.from_numpy(image.transpose(2,0,1).copy()).float()/255,torch.from_numpy(target)


def validate(net,loader,device):
    net.eval(); tp=pred=truth=0
    with torch.no_grad():
        for x,y in loader:
            p=net(x.to(device)).sigmoid().cpu()>.45
            g=y>.5
            tp+=(p&g).sum().item(); pred+=p.sum().item(); truth+=g.sum().item()
    return dict(dice=2*tp/max(1,pred+truth),precision=tp/max(1,pred),recall=tp/max(1,truth))


def evaluate_pitch(net,rows,device):
    errors=[]; good=0; samples=[]
    net.eval()
    with torch.no_grad():
        for i,row in enumerate(rows):
            x,_=LineDataset([row])[0]
            prob=net(x[None].to(device)).sigmoid()[0].cpu().numpy()
            h=homography_from_lines(heatmap_points(prob))
            if h is None:
                continue
            gt=[p for p in row["people"] if p["pitch"] is not None and p["role"]!="ball"]
            if not gt:
                continue
            # Labelled feet evaluate calibration alone. Detected feet will be
            # a separate end-to-end evaluation after candidate selection.
            feet=[((p["box"][0]+p["box"][2])/2/1920,p["box"][3]/1080) for p in gt]
            pred=project(h,feet)
            if not np.isfinite(pred).all():
                continue
            e=np.linalg.norm(pred-np.asarray([p["pitch"] for p in gt]),axis=1)
            errors.extend(e.tolist()); good+=1
            samples.append(dict(clip=row["clip"],image=row["image"],error_m=e.tolist(),H=h.tolist()))
    return dict(frames=len(rows),frames_with_calibration=good,players=len(errors),
                mean_m=float(np.mean(errors)) if errors else None,
                median_m=float(np.median(errors)) if errors else None,
                p90_m=float(np.percentile(errors,90)) if errors else None,
                note="Image-only learned marking calibration, labelled feet scoring; no player truth used in inference",
                samples=samples)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--prepare-only",action="store_true")
    ap.add_argument("--hours",type=float,default=1.2)
    ap.add_argument("--out",default="cv/out/training_20261011/pitch_lines")
    ap.add_argument("--device",default="mps")
    a=ap.parse_args(); root=Path(a.out)
    path=root/"data"/"rows.json"
    rows=json.loads(path.read_text()) if path.exists() else prepare(root/"data")
    print({k:len(v) for k,v in rows.items()},flush=True)
    if a.prepare_only:
        return
    torch.manual_seed(11026); np.random.seed(11026); torch.set_num_threads(2)
    device=torch.device(a.device)
    net=PitchLineNet().to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
    tr=DataLoader(LineDataset(rows["train"]),batch_size=12,shuffle=True,num_workers=0)
    va=DataLoader(LineDataset(rows["val"]),batch_size=12,num_workers=0)
    deadline=time.monotonic()+a.hours*3600
    history=[]; best=-1
    for ep in range(300):
        if time.monotonic()>=deadline:
            break
        net.train(); losses=[]
        for x,y in tr:
            if time.monotonic()>=deadline:
                break
            x,y=x.to(device),y.to(device)
            z=net(x); p=z.sigmoid()
            bce=torch.nn.functional.binary_cross_entropy_with_logits(z,y,pos_weight=torch.tensor(12.,device=device))
            dice=1-(2*(p*y).sum((2,3))+1)/(p.sum((2,3))+y.sum((2,3))+1)
            loss=bce+.3*dice.mean()
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        metrics=validate(net,va,device)
        history.append(dict(epoch=ep+1,loss=float(np.mean(losses)) if losses else None,**metrics))
        print(history[-1],flush=True)
        if metrics["dice"]>best:
            best=metrics["dice"]
            torch.save(dict(state={k:v.detach().cpu() for k,v in net.state_dict().items()},
                            names=NAMES,seed=11026,train_games=["4","6","9"],val_games=["3","5"],test_games=["2"]),root/"best.pt")
        (root/"history.json").write_text(json.dumps(history,indent=2))
        if len(history)>30 and np.argmax([h["dice"] for h in history])<len(history)-20:
            break
    net.load_state_dict(torch.load(root/"best.pt",weights_only=False)["state"])
    test=DataLoader(LineDataset(rows["test"]),batch_size=12,num_workers=0)
    report=dict(marking_test=validate(net,test,device),pitch_test=evaluate_pitch(net,rows["test"],device))
    (root/"results.json").write_text(json.dumps(report,indent=2))
    print({k:v if k!="pitch_test" else {a:b for a,b in v.items() if a!="samples"} for k,v in report.items()},flush=True)


if __name__=="__main__":
    main()
