import math
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, num_classes, vgg_cfg, batch_norm = False, **kwargs):
        super(VGG, self).__init__()
        self.feature_extractor = self._make_layers(vgg_cfg, batch_norm = batch_norm)
        dropout_rate = kwargs.get('dropout_rate', 0.5)
        self.final = nn.Sequential(
            Rearrange('b c h w -> b (c h w)'),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 512),
            nn.ReLU(inplace = True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 512),
            nn.ReLU(inplace = True),
            nn.Linear(512, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, math.sqrt(2. / (m.kernel_size[0] * m.kernel_size[1] * m.out_channels)))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.final(x)
        return x
    

    def loss(self, res, gt):
        return F.cross_entropy(res, gt)


    @staticmethod
    def _make_layers(cfg, batch_norm = False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels, v, kernel_size=3, padding=1))
                if batch_norm:
                    layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(inplace=True))
                in_channels = v
        return nn.Sequential(*layers)


def vgg11(num_classes, **kwargs):
    return VGG(num_classes, cfg['vgg11'], batch_norm = False, **kwargs)

def vgg11_bn(num_classes, **kwargs):
    return VGG(num_classes, cfg['vgg11'], batch_norm = True, **kwargs)

def vgg13(num_classes, **kwargs):
    return VGG(num_classes, cfg['vgg13'], batch_norm = False, **kwargs)

def vgg13_bn(num_classes, **kwargs):
    return VGG(num_classes, cfg['vgg13'], batch_norm = True, **kwargs)

def vgg16(num_classes, **kwargs):
    return VGG(num_classes, cfg['vgg16'], batch_norm = False, **kwargs)

def vgg16_bn(num_classes, **kwargs):
    return VGG(num_classes, cfg['vgg16'], batch_norm = True, **kwargs)

def vgg19(num_classes, **kwargs):
    return VGG(num_classes, cfg['vgg19'], batch_norm = False, **kwargs)

def vgg19_bn(num_classes, **kwargs):
    return VGG(num_classes, cfg['vgg19'], batch_norm = True, **kwargs)