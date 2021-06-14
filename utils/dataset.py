import torch
from torchvision import transforms
from torchvision.datasets import SVHN


def get_dataset(data_root = 'data/', split = 'train'):
    return SVHN(
        data_root,
        split,
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )