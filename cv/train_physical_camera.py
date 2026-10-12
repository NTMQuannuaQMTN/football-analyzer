"""Bounded training of an image-only physical camera from training labels."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader

from cv.camera_quality import assess_camera,plausible
from cv.physical_camera import PhysicalCameraNet,project


class Correspondences(Dataset):
    def __init__(self,rows,augment=False):self.rows=rows;self.augment=augment
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i];image=cv2.cvtColor(cv2.imread(row['image']),cv2.COLOR_BGR2RGB)
        x=torch.from_numpy(image.transpose(2,0,1).copy()).float()/255
        people=[p for p in row['people'] if p['pitch'] is not None and p['role'] in ('player','goalkeeper','referee')][:40]
        world=torch.zeros((40,2));feet=torch.zeros((40,2));mask=torch.zeros(40)
        for j,p in enumerate(people):world[j]=torch.tensor(p['pitch']);feet[j]=torch.tensor([(p['box'][0]+p['box'][2])/2/1920,p['box'][3]/1080]);mask[j]=1
        if self.augment:
            if torch.rand(())<.5:x=x.flip(-1);world[:,0]=105-world[:,0];feet[:,0]=1-feet[:,0]
            x=(x*(.85+.3*torch.rand(()))).clamp(0,1)
        return x,world,feet,mask


def validation(net,rows,device):
    errors=[];accepted=[];costs=[];support=0;net.eval()
    with torch.no_grad():
        for row in rows:
            x,world,feet,mask=Correspondences([row])[0]
            matrix,_=net(x[None].to(device));h=torch.linalg.inv(matrix)
            prediction=project(h,feet[None].to(device))[0].cpu().numpy()
            e=np.linalg.norm(prediction-world.numpy(),axis=1)[mask.numpy()>0]
            if not len(e):continue
            finite=np.isfinite(e).all();errors.extend(e.tolist() if finite else [])
            good=finite and plausible(assess_camera(cv2.imread(row['image']),h[0].cpu().numpy()),.75,.25)
            if good:accepted.extend(e.tolist());support+=1;costs.extend(np.minimum(e,20).tolist())
            else:costs.extend([20.]*len(e))
    return dict(total_frames=len(rows),accepted_frames=support,accepted_people=len(accepted),
        accepted_mean_m=float(np.mean(accepted)) if accepted else None,accepted_p90_m=float(np.percentile(accepted,90)) if accepted else None,
        unfiltered_mean_m=float(np.mean(errors)) if errors else None,selection_cost_m=float(np.mean(costs)),
        note='20m capped error/abstention selection loss; supported and unfiltered metre errors are separate')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--hours',type=float,default=2);ap.add_argument('--deadline',type=float)
    ap.add_argument('--device',default='mps');ap.add_argument('--out',default='cv/out/audit8_20261011/physical_camera')
    ap.add_argument('--resume');ap.add_argument('--world-weight',type=float,default=.05);a=ap.parse_args()
    if a.world_weight<0 or not np.isfinite(a.world_weight):raise ValueError('Invalid world-space loss weight')
    root=Path(a.out);root.mkdir(parents=True,exist_ok=True);torch.manual_seed(11026);np.random.seed(11026);torch.set_num_threads(1);cv2.setNumThreads(1)
    rows_path=Path('cv/out/training_20261011/pitch_lines/data/rows.json');rows=json.loads(rows_path.read_text())
    train_rows=[r for r in rows['train'] if sum(p['pitch'] is not None and p['role'] in ('player','goalkeeper','referee') for p in r['people'])>=7]
    net=PhysicalCameraNet().to(a.device);cached=Path('/Users/qnrj/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth')
    if not cached.is_file():raise FileNotFoundError('Local ImageNet weights missing; no download attempted')
    state={k:v for k,v in torch.load(cached,weights_only=True).items() if not k.startswith('fc.')};net.encoder.load_state_dict(state,strict=False)
    resume_digest=None
    if a.resume:
        resume=Path(a.resume)
        if resume.resolve()==(root/'best.pt').resolve():raise ValueError('Resume into a new output directory')
        resume_digest=hashlib.sha256(resume.read_bytes()).hexdigest()
        net.load_state_dict(torch.load(resume,weights_only=False)['state'])
    config=dict(architecture='physical_camera_resnet18',seed=11026,device=a.device,hours=a.hours,deadline=a.deadline,
        rows_sha256=hashlib.sha256(rows_path.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        camera_source_sha256=hashlib.sha256(Path('cv/physical_camera.py').read_bytes()).hexdigest(),
        train_games=['4','6','9'],validation_games=['3','5'],test_game='2',test_evaluated=False,training_frames=len(train_rows),
        training_targets='Visible ground points/box feet supervise camera; image only at inference',
        resume=a.resume,resume_sha256=resume_digest,world_loss_weight=a.world_weight,
        selection='Fixed image-only plausibility gate and 20m capped validation error/abstention cost on every fourth frame',
        limitations='Main sideline camera bounds; no guarantee for reverse angles, close-ups or different stadium camera locations')
    (root/'config.json').write_text(json.dumps(config,indent=2));(root/'training_source.py').write_bytes(Path(__file__).read_bytes())
    started=time.time();deadline=min(started+a.hours*3600,a.deadline or float('inf'))
    loader=DataLoader(Correspondences(train_rows,True),batch_size=8,shuffle=True,num_workers=0)
    optimizer=torch.optim.AdamW(net.parameters(),lr=.00005,weight_decay=.0001);history=[];best=float('inf');last_best=0
    if a.resume:
        baseline=validation(net,rows['val'][::4],a.device);best=baseline['selection_cost_m']
        history.append(dict(epoch=0,train_loss=None,validation=baseline,condition='Frozen resume checkpoint before updates'))
        torch.save(dict(state={k:v.detach().cpu() for k,v in net.state_dict().items()},**config,validation=baseline),root/'best.pt')
    for epoch in range(250):
        if time.time()>=deadline:break
        net.train()
        for layer in net.encoder.modules():
            if isinstance(layer,nn.BatchNorm2d):layer.eval()
        losses=[]
        for image,world,feet,mask in loader:
            if time.time()>=deadline:break
            image,world,feet,mask=[v.to(a.device) for v in [image,world,feet,mask]]
            matrix,pose=net(image);image_prediction=project(matrix,world)
            loss=(nn.functional.smooth_l1_loss(image_prediction,feet,reduction='none',beta=.01).sum(-1)*mask).sum()/mask.sum().clamp_min(1)
            if epoch>=5 or a.resume:
                inverse_prediction=project(torch.linalg.inv(matrix),feet)/torch.tensor([105.,68.],device=a.device)
                world_loss=nn.functional.smooth_l1_loss(inverse_prediction.clamp(-.2,1.2),world/torch.tensor([105.,68.],device=a.device),reduction='none',beta=.01).sum(-1)
                loss+=a.world_weight*(world_loss*mask).sum()/mask.sum().clamp_min(1)
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5.);optimizer.step();losses.append(loss.item())
        row=dict(epoch=epoch+1,train_loss=float(np.mean(losses)) if losses else None,
            optimizer_updates=len(losses),complete_training_epoch=len(losses)==len(loader))
        if epoch%5==0:
            row['validation']=validation(net,rows['val'][::4],a.device);cost=row['validation']['selection_cost_m']
            if cost<best:
                best=cost;last_best=epoch
                torch.save(dict(state={k:v.detach().cpu() for k,v in net.state_dict().items()},**config,validation=row['validation']),root/'best.tmp.pt');(root/'best.tmp.pt').replace(root/'best.pt')
        history.append(row);(root/'history.json').write_text(json.dumps(history,indent=2));print(row,flush=True)
        if epoch-last_best>=40:break
    (root/'finished.json').write_text(json.dumps(dict(started_utc_epoch=started,finished_utc_epoch=time.time(),epochs=sum(r['epoch']>0 for r in history),complete_epochs=sum(r.get('complete_training_epoch',False) for r in history),optimizer_updates=sum(r.get('optimizer_updates',0) for r in history),validation_best_cost=best,test_evaluated=False),indent=2));print('Finished physical camera experiment',flush=True)


if __name__=='__main__':main()
