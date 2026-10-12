"""Anonymous appearance embedding from existing local person crops.

Embeddings aid association; they do not verify names or read shirt numbers.
"""
import torch
from torch import nn


class PersonEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18
        self.encoder=resnet18(weights=None);self.encoder.fc=nn.Identity()
        self.projection=nn.Linear(512,128)

    def forward(self,x):return nn.functional.normalize(self.projection(self.encoder(x)),dim=1)


def batch_hard_loss(z,labels,margin=.2):
    distance=1-z@z.T
    same=labels[:,None]==labels[None,:]
    positive=same&~torch.eye(len(z),device=z.device,dtype=torch.bool)
    negative=~same
    valid=positive.any(1)&negative.any(1)
    if not valid.any():return z.sum()*0
    hardest_positive=distance.masked_fill(~positive,-float('inf')).max(1).values
    hardest_negative=distance.masked_fill(~negative,float('inf')).min(1).values
    return nn.functional.relu(hardest_positive[valid]-hardest_negative[valid]+margin).mean()
