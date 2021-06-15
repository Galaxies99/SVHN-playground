import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .binary2categorical import Binary2Categorical
from einops import rearrange


class LogisticRegression(nn.Module):
    def __init__(self, **kwargs):
        super(LogisticRegression, self).__init__()
        dim = kwargs.get('dim', 32 * 32 * 3)
        self.final = nn.Linear(dim, 1, bias = False)
    
    def forward(self, x):
        if x.dim() == 3:
            x = rearrange(x, 'b h w -> b (h w)')
        elif x.dim() == 4:
            x = rearrange(x, 'b c h w -> b (c h w)')
        return torch.sigmoid(self.final(x).view(-1))
    
    def loss(self, pred, gt, balance = 1.0):
        negative_log_likelihood = - pred * gt + torch.log(1.0 + torch.exp(pred))
        positive_samples_nll = torch.where(gt == 1, negative_log_likelihood, torch.scalar_tensor(0.0, dtype = torch.float32).to(pred.device)).to(pred.device)
        total_samples = len(negative_log_likelihood) + len(positive_samples_nll) * (balance - 1.0)
        return (positive_samples_nll.sum() * (balance - 1.0) + negative_log_likelihood.sum()) / total_samples
       

class CategoricalLogisticRegression(Binary2Categorical):
    def __init__(self, **kwargs):
        super(CategoricalLogisticRegression, self).__init__(model = LogisticRegression, **kwargs)

