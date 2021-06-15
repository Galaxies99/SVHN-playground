import torch
import torch.nn as nn
import torch.nn.functional as F


class Binary2Categorical(nn.Module):
    def __init__(self, model, num_classes, **kwargs):
        super(Binary2Categorical, self).__init__()
        self.num_classes = num_classes
        models = []
        for _ in range(self.num_classes):
            models.append(model(**kwargs))
        self.models = nn.ModuleList(models)
        self.loss_type = kwargs.get('loss_type', 'default')
        if self.loss_type not in ['default', 'cross_entropy', 'binary_cross_entropy']:
            raise NotImplementedError("Invalid loss type.")
    
    def forward(self, x):
        res = []
        for i in range(self.num_classes):
            res.append(self.models[i](x).view(-1, 1))
        return torch.cat(res, dim = 1)
    
    def loss(self, pred, gt):
        return F.cross_entropy(pred, gt)
