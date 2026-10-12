"""Bounded local camera training; select by validation geometry and coverage.

Local ImageNet weights only. Existing exports/checkpoints are never replaced.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader

from cv.gsr import project
from cv.pitch_lines import NAMES,PitchLineNet,heatmap_points,homography_from_lines
from cv.train_pitch_lines import LineDataset,validate


class ResNetPitch(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18
        self.encoder=resnet18(weights=None)
        def block(a,b):return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(8,b),nn.SiLU(),nn.Conv2d(b,b,3,padding=1),nn.GroupNorm(8,b),nn.SiLU())
        self.up3=block(768,128);self.up2=block(256,96);self.up1=block(160,64);self.head=nn.Conv2d(64,len(NAMES),1)
        self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])

    def forward(self,x):
        e=self.encoder;x=(x-self.mean)/self.std
        x=e.maxpool(e.relu(e.bn1(e.conv1(x))));a=e.layer1(x);b=e.layer2(a);c=e.layer3(b);d=e.layer4(c)
        up=lambda x,y:torch.nn.functional.interpolate(x,size=y.shape[-2:],mode='bilinear',align_corners=False)
        z=self.up3(torch.cat([up(d,c),c],1));z=self.up2(torch.cat([up(z,b),b],1));z=self.up1(torch.cat([up(z,a),a],1))
        return self.head(z)


class AugmentedLines(Dataset):
    def __init__(self,rows):self.base=LineDataset(rows)
    def __len__(self):return len(self.base)
    def __getitem__(self,i):
        x,y=self.base[i]
        if torch.rand(())<.5:
            swap=lambda n:n.replace('left','TEMP').replace('right','left').replace('TEMP','right')
            indexes=[NAMES.index(swap(n)) for n in NAMES]
            x=x.flip(-1);y=y[indexes].flip(-1)
        # Photometric only; image/pitch geometry remains unchanged.
        brightness=.8+.4*torch.rand(());contrast=.85+.3*torch.rand(())
        x=((x-.5)*contrast+.5)*brightness
        return x.clamp(0,1),y


def geometry_score(net,rows,device):
    """All labelled people contribute: missing camera gets a 20 m selection cost.

    The capped score selects checkpoints only, never reported as position MAE.
    Missing-camera counts and uncapped supported errors are separate.
    """
    costs=[];errors=[];supported=0;net.eval()
    with torch.no_grad():
        for row in rows:
            x,_=LineDataset([row])[0];prob=net(x[None].to(device)).sigmoid()[0].cpu().numpy()
            h=homography_from_lines(heatmap_points(prob))
            people=[p for p in row['people'] if p['pitch'] is not None and p['role'] in ('player','goalkeeper','referee')]
            if not people:continue
            feet=[((p['box'][0]+p['box'][2])/2/1920,p['box'][3]/1080) for p in people]
            if h is None:costs.extend([20.]*len(people));continue
            error=np.linalg.norm(project(h,feet)-np.array([p['pitch'] for p in people]),axis=1)
            if not np.isfinite(error).all():costs.extend([20.]*len(people));continue
            supported+=1;errors.extend(error.tolist());costs.extend(np.minimum(error,20).tolist())
    return dict(selection_cost_m=float(np.mean(costs)),supported_frames=supported,total_frames=len(rows),
                supported_people=len(errors),supported_mean_m=float(np.mean(errors)) if errors else None,
                supported_median_m=float(np.median(errors)) if errors else None,
                supported_p90_m=float(np.percentile(errors,90)) if errors else None,
                note='20 m capped per-person error/abstention penalty is a validation selection criterion, not position accuracy')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arch',choices=['tiny','resnet18'],default='resnet18')
    ap.add_argument('--rows',default='cv/out/audit8_20261011/camera_training/rows.json')
    ap.add_argument('--hours',type=float,default=2);ap.add_argument('--device',default='mps')
    ap.add_argument('--out',required=True);ap.add_argument('--initial')
    ap.add_argument('--deadline',type=float,help='Absolute UTC epoch cutoff; protects the session across host sleep')
    ap.add_argument('--epochs',type=int,default=300,help='Maximum epochs, still bounded by --hours and --deadline')
    ap.add_argument('--geometry-every',type=int,default=5,help='Validation geometry interval in epochs')
    a=ap.parse_args();root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
    if a.epochs<1 or a.geometry_every<1 or not np.isfinite(a.hours) or a.hours<=0:raise ValueError('Positive finite training budget required')
    if a.deadline is not None and (not np.isfinite(a.deadline) or a.deadline<=time.time()):raise ValueError('Future finite deadline required')
    if (root/'config.json').exists():raise ValueError('Use a fresh output directory; do not overwrite an experiment')
    torch.manual_seed(11026);np.random.seed(11026);cv2.setRNGSeed(11026);torch.set_num_threads(1);cv2.setNumThreads(1)
    rows_path=Path(a.rows);rows=json.loads(rows_path.read_text());net=(ResNetPitch() if a.arch=='resnet18' else PitchLineNet()).to(a.device)
    pretrained=None
    if a.initial:net.load_state_dict(torch.load(a.initial,weights_only=False)['state'])
    elif a.arch=='resnet18':
        pretrained=Path('/Users/qnrj/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth')
        if not pretrained.exists():raise FileNotFoundError('Local pretrained weights missing; no download attempted')
        net.encoder.load_state_dict(torch.load(pretrained,weights_only=True))
    config=dict(architecture=a.arch,seed=11026,rows_sha256=hashlib.sha256(rows_path.read_bytes()).hexdigest(),
        train_games=['4','6','9'],validation_games=['3','5'],test_game='2',test_evaluated=False,
        initial=a.initial,initial_sha256=hashlib.sha256(Path(a.initial).read_bytes()).hexdigest() if a.initial else None,
        pretrained=str(pretrained) if pretrained else None,hours=a.hours,device=a.device,max_epochs=a.epochs,deadline_epoch=a.deadline,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        geometry_every=a.geometry_every,
        selection='Validation coverage-penalised capped geometry error on every fourth validation image',
        augmentation='Consistent horizontal reflection of images and named pitch lines; brightness/contrast only')
    (root/'config.json').write_text(json.dumps(config,indent=2))
    (root/'training_source.py').write_bytes(Path(__file__).read_bytes())
    if a.arch=='resnet18':
        encoder=list(net.encoder.parameters());ids={id(p) for p in encoder};decoder=[p for p in net.parameters() if id(p) not in ids]
        optimizer=torch.optim.AdamW([{'params':encoder,'lr':.00005},{'params':decoder,'lr':.0005}],weight_decay=.0001)
    else:optimizer=torch.optim.AdamW(net.parameters(),lr=.0002,weight_decay=.0001)
    train=DataLoader(AugmentedLines(rows['train']),batch_size=8,shuffle=True,num_workers=0)
    val=DataLoader(LineDataset(rows['val']),batch_size=8,num_workers=0)
    deadline=min(time.time()+a.hours*3600,a.deadline or float('inf'));history=[];best=float('inf');last_best=0
    if a.initial:
        baseline=geometry_score(net,rows['val'][::4],a.device);best=baseline['selection_cost_m']
        history.append(dict(epoch=0,geometry=baseline,condition='Frozen initial checkpoint before updates'))
        torch.save(dict(state={k:v.detach().cpu() for k,v in net.state_dict().items()},**config,validation_geometry=baseline),root/'best.pt')
        (root/'history.json').write_text(json.dumps(history,indent=2));print(history[-1],flush=True)
    for epoch in range(a.epochs):
        if time.time()>=deadline:break
        net.train()
        if a.arch=='resnet18':
            # Keep pretrained running statistics stable on small broadcast batches.
            for layer in net.encoder.modules():
                if isinstance(layer,nn.BatchNorm2d):layer.eval()
        losses=[]
        for x,y in train:
            if time.time()>=deadline:break
            x,y=x.to(a.device),y.to(a.device);z=net(x);p=z.sigmoid()
            loss=nn.functional.binary_cross_entropy_with_logits(z,y,pos_weight=torch.tensor(12.,device=a.device))
            loss+=.3*(1-(2*(p*y).sum((2,3))+1)/(p.sum((2,3))+y.sum((2,3))+1)).mean()
            optimizer.zero_grad();loss.backward();optimizer.step();losses.append(loss.item())
        row=dict(epoch=epoch+1,train_loss=float(np.mean(losses)) if losses else None,
                 training_batches=len(losses),complete_training_epoch=len(losses)==len(train),**validate(net,val,a.device))
        if epoch%a.geometry_every==0:
            row['geometry']=geometry_score(net,rows['val'][::4],a.device)
            value=row['geometry']['selection_cost_m']
            if value<best:
                best=value;last_best=epoch
                torch.save(dict(state={k:v.detach().cpu() for k,v in net.state_dict().items()},**config,validation_geometry=row['geometry']),root/'best.pt')
        history.append(row);(root/'history.json').write_text(json.dumps(history,indent=2));print(row,flush=True)
        if epoch-last_best>=35:break
    (root/'finished.json').write_text(json.dumps(dict(completed_epochs=sum(r['epoch']>0 for r in history),
        finished_epoch=time.time(),best_selection_cost_m=best,test_evaluated=False),indent=2))
    print('Completed training; test remains unevaluated',sum(r['epoch']>0 for r in history),'epochs',flush=True)


if __name__=='__main__':main()
