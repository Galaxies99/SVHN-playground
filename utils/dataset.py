import os
import cv2
import torch
import numpy as np
from scipy import io
from torch.utils.data import Dataset
from .hog_feature import HOG_Feature_Descriptor
import logging
from .logger import ColoredLogger


logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger(__name__)


class SVHN_Dataset(Dataset):
    def __init__(self, data_root, split = 'train', gray_scale = False, hog_feature = False, preprocess = False, **kwargs):
        super(SVHN_Dataset, self).__init__()
        if split not in ['train', 'test', 'extra']:
            raise AttributeError('Invalid "split" attribute, "split" can only be "train", "test", or "extra".')
        self.preprocess = preprocess

        data = io.loadmat(os.path.join(data_root, split + "_32x32.mat"))
        self.images = data['X'].transpose(3, 0, 1, 2)
        self.images = self.images[:, :, :, ::-1] 
        self.hog_feature = hog_feature
        self.gray_scale = gray_scale
        if self.hog_feature:
            self.block_size = kwargs.get('block_size', 2)
            self.cell_size = kwargs.get('cell_size', 2)
            self.bin_size = kwargs.get('bin_size', 9)
            self.hog = HOG_Feature_Descriptor(
                image_size = (32, 32), 
                pixels_per_cell = (self.cell_size, self.cell_size),
                cells_per_block = (self.block_size, self.block_size),
                bin_size = self.bin_size
            )
        if self.hog_feature and self.gray_scale:
            prepared_feature_file = os.path.join(data_root, 'feature_{}_{}_{}_{}.npy'.format(split, self.block_size, self.cell_size, self.bin_size))
            prepared_label_file = os.path.join(data_root, 'label_{}_{}_{}_{}.npy'.format(split, self.block_size, self.cell_size, self.bin_size))
            if os.path.exists(prepared_feature_file) and os.path.exists(prepared_label_file):
                logger.info('Use existing features.')
                self.images = np.load(prepared_feature_file)
                self.labels = np.load(prepared_label_file)
                self.preprocess = True
                return 
        if self.preprocess:
            if self.gray_scale:
                self.images = self._rgb2gray_batch(self.images)
            if self.hog_feature:
                if self.gray_scale:
                    self.images = self.hog.batch_gray_extractor(self.images)
                else:
                    self.images = self.hog.batch_RGB_extractor(self.images)
            else:
                self.images = self._transform(self.images)
        self.labels = data['y']
        self.labels = np.where(self.labels == 10, 0, self.labels)
        
    def _transform(self, image):
        return image / 255.0 * 2 - 1
    
    def _rgb2gray_batch(self, images):
        R = images[:, :, :, 0]
        G = images[:, :, :, 1]
        B = images[:, :, :, 2]
        return R * 0.299 + G * 0.587 + B * 0.114

    def _rgb2gray(self, images):
        R = images[:, :, 0]
        G = images[:, :, 1]
        B = images[:, :, 2]
        return R * 0.299 + G * 0.587 + B * 0.114

    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        if not self.preprocess:
            if self.gray_scale:
                image = self._rgb2gray(image)
            if self.hog_feature:
                if self.gray_scale:
                    image = self.hog.gray_extractor(image)
                else:
                    image = self.hog.RGB_extractor(image)
            else:
                image = self._transform(image)
        return torch.FloatTensor(image), torch.LongTensor(label)


def get_dataset_size(data_root = 'data', split = 'train'):
    return len(SVHN_Dataset(data_root, split))


def generate_feature(path = 'data', split = 'train', gray_scale = True, block_size = 2, cell_size = 2, bin_size = 9):
    d = SVHN_Dataset(path, split, gray_scale, hog_feature = True, preprocess = True, block_size = block_size, cell_size = cell_size, bin_size = bin_size)
    filename = os.path.join(path, 'feature_{}_{}_{}_{}.npy'.format(split, block_size, cell_size, bin_size))
    label_filename = os.path.join(path, 'label_{}_{}_{}_{}.npy'.format(split, block_size, cell_size, bin_size))
    np.save(filename, d.images)
    np.save(label_filename, d.labels)