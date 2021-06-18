import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class AlexNet(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(AlexNet, self).__init__()
        self.dropout_rate = kwargs.get('dropout_rate', 0.5)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 3, stride = 2, padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            nn.Conv2d(64, 192, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            nn.Conv2d(192, 384, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(384, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
        )
        self.final = nn.Sequential(
            Rearrange('b c h w -> b (c h w)'),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256 * 1 * 1, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Linear(4096, num_classes),
        )


    def forward(self, x):
        x = self.features(x)
        return self.final(x)
    
    
    def loss(self, res, gt):
        return F.cross_entropy(res, gt)