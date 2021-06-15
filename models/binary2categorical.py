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
        if self.loss_type not in ['default', 'cross_entropy']:
            raise NotImplementedError("Invalid loss type.")
    
    def forward(self, x):
        res = []
        for i in range(self.num_classes):
            res.append(self.models[i](x).view(1, -1))
        return torch.cat(res, dim = 0)
    
    def loss(self, pred, gt):
        if self.loss_type == 'default':
            return self.default_loss(pred, gt)
        elif self.loss_type == 'cross_entropy':
            return self.cross_entropy_loss(pred, gt)
        
    def default_loss(self, pred, gt):
        losses = []
        for i in range(self.num_classes):
            gt_i = torch.FloatTensor([(1 if gt_s == i else 0) for gt_s in gt]).to(pred.device)
            losses.append(self.models[i].loss(pred[i], gt_i).view(-1))
        return torch.cat(losses, dim = 0), True

    def cross_entropy_loss(self, pred, gt):
        return F.cross_entropy(pred.transpose(1, 0), gt.view(-1)), False
