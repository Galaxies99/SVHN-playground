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
        self.beta = nn.Parameter(torch.FloatTensor(dim))
    
    def forward(self, x):
        x = rearrange(x, 'b c h w -> b (c h w)')
        return torch.sigmoid(torch.matmul(x, self.beta.view(-1, 1)).view(-1))
    
    def loss(self, pred, gt):
        return (- pred * gt + torch.log(1.0 + torch.exp(pred))).mean()

class CategoricalLogisticRegression(Binary2Categorical):
    def __init__(self, **kwargs):
        super(CategoricalLogisticRegression, self).__init__(model = LogisticRegression, **kwargs)

