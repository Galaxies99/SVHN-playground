import os
import cv2
import torch
import numpy as np
from scipy import io
from torch.utils.data import Dataset
from .hog_feature import calc_hog


class SVHN_Dataset(Dataset):
    def __init__(self, data_root, split = 'train', gray_scale = False, hog_feature = False, **kwargs):
        super(SVHN_Dataset, self).__init__()
        if split not in ['train', 'test', 'extra']:
            raise AttributeError('Invalid "split" attribute, "split" can only be "train", "test", or "extra".')
        if not gray_scale and hog_feature:
            raise AttributeError('hog feature extraction must be executed under grayscale mode.')
        data = io.loadmat(os.path.join(data_root, split + "_32x32.mat"))
        self.images = data['X']
        self.labels = data['y']
        self.gray_scale = gray_scale
        self.hog_feature = hog_feature
        self.block_size = kwargs.get('block_size', 2)
        self.cell_size = kwargs.get('cell_size', 2)
        self.bin_size = kwargs.get('bin_size', 9)
    
    def _transform(self, image):
        return image / 255.0 * 2 - 1

    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, index):
        image = self.images[:, :, :, index]
        label = self.labels[index]
        if self.gray_scale:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            if self.hog_feature:
                image = calc_hog(image, block_size = self.block_size, cell_size = self.cell_size, bin_size = self.bin_size)
            else:
                image = image[np.newaxis, :, :]
        else:
            image = image.transpose(2, 0, 1)
        if not self.hog_feature:
            image = self._transform(image)
        label = np.where(label == 10, 0, label)
        return torch.FloatTensor(image), torch.LongTensor(label)


def get_dataset(data_root = 'data', split = 'train', gray_scale = False, hog_feature = False):
    return SVHN_Dataset(data_root, split, gray_scale = gray_scale, hog_feature = hog_feature)


def get_dataset_size(data_root = 'data', split = 'train'):
    return len(get_dataset(data_root, split))