import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class LeNet(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(LeNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size = 5),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(6, 16, kernel_size = 5),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2)
        )

        self.final = nn.Sequential(
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace = True),
            nn.Linear(120, 84),
            nn.ReLU(inplace = True),
            nn.Linear(84, num_classes)
        )
    

    def forward(self, x):
        x = self.conv(x)
        return self.final(x)
    
    
    def loss(self, res, gt):
        return F.cross_entropy(res, gt)