import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride = 1):
        super(BasicBlock, self).__init__()

        resblock_out_channels = out_channels * BasicBlock.expansion
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, resblock_out_channels, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(resblock_out_channels)
        )

        if stride == 1 and in_channels == resblock_out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, resblock_out_channels, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(resblock_out_channels),
            )
        
        self.final = nn.ReLU(inplace = True)


    def forward(self, x):
        return self.final(self.residual(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride = 1):
        super().__init__()

        resblock_out_channels = out_channels * BottleNeck.expansion
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, stride = stride, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, resblock_out_channels, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(resblock_out_channels)
        )

        if stride == 1 and in_channels == resblock_out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, resblock_out_channels, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(resblock_out_channels),
            )

        self.final = nn.ReLU(inplace = True)

    
    def forward(self, x):
        return self.final(self.residual(x) + self.shortcut(x))



class ResNet(nn.Module):
    def __init__(self, block_type, num_block, num_classes, **kwargs):
        super(ResNet, self).__init__()

        if block_type not in ['BasicBlock', 'BottleNeck']:
            raise NotImplementedError('Invalid block type.')
        
        block = BasicBlock if block_type == 'BasicBlock' else BottleNeck
        self.num_classes = num_classes

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride = 1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.cur_channels = 64

        self.layer2 = self._make_layer(block, 64, num_block[0], 1)
        self.layer3 = self._make_layer(block, 128, num_block[1], 2)
        self.layer4 = self._make_layer(block, 256, num_block[2], 2)
        self.layer5 = self._make_layer(block, 512, num_block[3], 2)

        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(512 * block.expansion, num_classes)
        )
    

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            cur_stride = stride if i == 0 else 1
            layers.append(block(self.cur_channels, out_channels, cur_stride))
            self.cur_channels = out_channels * block.expansion
        return nn.Sequential(*layers)
    

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return self.final(x)
    
    
    def loss(self, res, gt):
        return F.cross_entropy(res, gt)


def resnet18(**kwargs):
    return ResNet('BasicBlock', [2, 2, 2, 2], **kwargs)

def resnet34(**kwargs):
    return ResNet('BasicBlock', [3, 4, 6, 3], **kwargs)

def resnet50(**kwargs):
    return ResNet('BottleNeck', [3, 4, 6, 3], **kwargs)

def resnet101(**kwargs):
    return ResNet('BottleNeck', [3, 4, 23, 3], **kwargs)

def resnet152(**kwargs):
    return ResNet('BottleNeck', [3, 8, 36, 3], **kwargs)

def resnet(**kwargs):
    return ResNet(**kwargs)