import os
import yaml
import argparse
import logging
import numpy as np
from tqdm import tqdm
from utils.logger import ColoredLogger
from sklearn.svm import SVC
import argparse


logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', '-c', default = os.path.join('configs', 'SVM', 'default.yaml'), help = 'Config File', type = str)
FLAGS = parser.parse_args()
CFG_FILE = FLAGS.cfg

with open(CFG_FILE, 'r') as cfg_file:
    cfg_dict = yaml.load(cfg_file, Loader=yaml.FullLoader)

feature_path = cfg_dict.get('train_feature_path', os.path.join('data', 'hog-features.npy'))
label_path = cfg_dict.get('train_label_path', os.path.join('data', 'labels.npy'))

test_feature_path = cfg_dict.get('test_feature_path', os.path.join('data', 'hog-features.npy'))
test_label_path = cfg_dict.get('test_label_path', os.path.join('data', 'labels.npy'))

samples = cfg_dict.get('samples', 10000)
C = cfg_dict.get('regularization', 100000)
kernel = cfg_dict.get('kernel', 'linear')
degree = cfg_dict.get('degree', 3)
gamma = cfg_dict.get('gamma', 'scale')
coef0 = cfg_dict.get('coefficient', 0.0)

images = np.load(feature_path)
labels = np.load(label_path)

images = images[:samples]
labels = labels[:samples]
labels = labels.reshape(-1)

model = SVC(C = C, kernel = kernel, degree = degree, gamma = gamma, coef0 = coef0, verbose = True)
model.fit(images, labels)

test_images = np.load(test_feature_path)
test_labels = np.load(test_label_path)
test_labels = test_labels.reshape(-1)

logger.info('Acc: {:.6f}', model.score(test_images, test_labels)[0])
