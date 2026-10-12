import torch

from cv.person_embedding import batch_hard_loss


def test_triplet_loss_rewards_identity_separation_and_handles_singletons():
    labels=torch.tensor([0,0,1,1])
    good=torch.tensor([[1.,0],[1.,0],[0,1.],[0,1.]])
    collapsed=torch.tensor([[1.,0]]*4)
    assert batch_hard_loss(good,labels)==0
    assert batch_hard_loss(collapsed,labels)>0
    assert batch_hard_loss(good,torch.arange(4))==0
