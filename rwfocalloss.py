import torch.nn.functional as F
import torch
import torch.nn as nn


class MultFocalLoss(nn.Module):
    def __init__(self,  alpha=0.5, gamma=1, reduction="mean"):
        super(MultFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logit, target):
        prob = F.softmax(logit, dim=1)

        #ori_shape = target.shape
        target = target.view(-1, 1)
        prob = prob.gather(1, target).view(-1)
        log_pt = torch.log(prob)
        loss = -self.alpha*torch.pow(torch.sub(1.0, prob), self.gamma)*log_pt
        loss = loss.mean()

        return loss
