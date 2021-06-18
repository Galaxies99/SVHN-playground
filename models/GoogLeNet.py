import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class Inception(nn.Module):
    def __init__(self, in_channels, n1, n3c, n3, n5c, n5, pool_c):
        super(Inception, self).__init__()

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, n1, kernel_size = 1),
            nn.BatchNorm2d(n1),
            nn.ReLU(inplace = True)
        )

        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, n3c, kernel_size = 1),
            nn.BatchNorm2d(n3c),
            nn.ReLU(inplace = True),
            nn.Conv2d(n3c, n3, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(n3),
            nn.ReLU(inplace = True)
        )

        self.conv5x5 = nn.Sequential(
            nn.Conv2d(in_channels, n5c, kernel_size = 1),
            nn.BatchNorm2d(n5c),
            nn.ReLU(inplace = True),
            nn.Conv2d(n5c, n5, kernel_size = 5, stride = 1, padding = 2),
            nn.BatchNorm2d(n5),
            nn.ReLU(inplace = True)
        )

        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1),
            nn.Conv2d(in_channels, pool_c, kernel_size = 1),
            nn.BatchNorm2d(pool_c),
            nn.ReLU(inplace = True)
        )


    def forward(self, x):
        return torch.cat([self.conv1x1(x), self.conv3x3(x), self.conv5x5(x), self.pool(x)], dim = 1)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(GoogLeNet, self).__init__()

        self.num_classes = num_classes
        self.dropout_rate = kwargs.get('dropout_rate', 0.4)

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True), 
            nn.Conv2d(64, 192, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )

        self.layer2 = nn.Sequential(
            Inception(192, 64, 96, 128, 16, 32, 32),
            Inception(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )
        
        self.layer3 = nn.Sequential(
            Inception(480, 192, 96, 208, 16, 48, 64),
            Inception(512, 160, 112, 224, 24, 64, 64),
            Inception(512, 128, 128, 256, 24, 64, 64),
            Inception(512, 112, 144, 288, 32, 64, 64),
            Inception(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )

        self.layer4 = nn.Sequential(
            Inception(832, 256, 160, 320, 32, 128, 128),
            Inception(832, 384, 192, 384, 48, 128, 128)
        )

        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(p = self.dropout_rate),
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(1024, num_classes)
        )


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.final(x)


    def loss(self, res, gt):
        return F.cross_entropy(res, gt)
