"""Image-only camera model constrained to a ground-plane pinhole camera.

Bounds describe a main sideline camera, not every possible broadcast shot.
Training correspondence labels are never inputs to this network.
"""
import torch
from torch import nn

BOUNDS=((35.,70.),(73.,125.),(8.,50.),(-10.,115.),(-5.,73.),(.4,12.),(-.12,.12))


def camera_matrix(pose,aspect=16/9):
    """World metres -> normalised image; pose=(cx,cy,cz,aimx,aimy,focal,roll)."""
    camera=pose[...,:3]
    target=torch.cat([pose[...,3:5],torch.zeros_like(pose[...,:1])],-1)
    forward=nn.functional.normalize(target-camera,dim=-1)
    vertical=torch.zeros_like(forward);vertical[...,2]=1
    right=nn.functional.normalize(torch.linalg.cross(vertical,forward,dim=-1),dim=-1)
    up=torch.linalg.cross(forward,right,dim=-1);down=-up
    cosine=pose[...,6:7].cos();sine=pose[...,6:7].sin()
    rotation=torch.stack([cosine*right+sine*down,-sine*right+cosine*down,forward],-2)
    translation=-(rotation@camera[...,None])[...,0]
    plane=torch.cat([rotation[...,:,:2],translation[...,None]],-1)
    intrinsics=torch.zeros_like(plane);intrinsics[...,0,0]=pose[...,5];intrinsics[...,1,1]=pose[...,5]*aspect
    intrinsics[...,0,2]=.5;intrinsics[...,1,2]=.5;intrinsics[...,2,2]=1
    return intrinsics@plane


def project(matrix,points):
    ones=torch.ones_like(points[...,:1]);hp=torch.cat([points,ones],-1)@matrix.transpose(-1,-2)
    denominator=hp[...,2:3]
    denominator=torch.where(denominator.abs()<1e-6,torch.where(denominator>=0,1e-6,-1e-6),denominator)
    return hp[...,:2]/denominator


class PhysicalCameraNet(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18
        self.encoder=resnet18(weights=None);self.encoder.fc=nn.Linear(512,7)
        low=torch.tensor([p[0] for p in BOUNDS]);high=torch.tensor([p[1] for p in BOUNDS])
        self.register_buffer('low',low);self.register_buffer('high',high)
        self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None]);self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])
        initial=torch.tensor([52.5,95.,22.,52.5,34.,1.5,0.]);fraction=(initial-low)/(high-low)
        nn.init.zeros_(self.encoder.fc.weight)
        with torch.no_grad():self.encoder.fc.bias.copy_(torch.logit(fraction))

    def forward(self,image):
        pose=self.low+(self.high-self.low)*self.encoder((image-self.mean)/self.std).sigmoid()
        return camera_matrix(pose),pose
