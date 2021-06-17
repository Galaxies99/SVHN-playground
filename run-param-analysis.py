import os
import yaml
import torch
import argparse
import logging
from utils.logger import ColoredLogger
from utils.builder import categorical_model_builder
import numpy as np
import argparse


logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', '-c', default = os.path.join('configs', 'VAE', 'default.yaml'), help = 'Config File', type = str)
FLAGS = parser.parse_args()
CFG_FILE = FLAGS.cfg

with open(CFG_FILE, 'r') as cfg_file:
    cfg_dict = yaml.load(cfg_file, Loader=yaml.FullLoader)


model_params = cfg_dict.get('model', {})
trainer_params = cfg_dict.get('trainer', {})
stats_params = cfg_dict.get('stats', {})

logger.info('Building models ...')
num_classes = trainer_params.get('num_classes', 10)
models = []
for _ in range(num_classes):
    models.append(categorical_model_builder(model_params))

logger.info('Checking checkpoints ...')
stats_dir = os.path.join(stats_params.get('stats_dir', 'stats'), stats_params.get('stats_folder', 'temp'))
if os.path.exists(stats_dir) == False:
    os.makedirs(stats_dir)
checkpoint_file = os.path.join(stats_dir, 'checkpoint.tar')
if os.path.isfile(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    for i in range(num_classes):
        models[i].load_state_dict(checkpoint['model_state_dict'][i])
    start_epoch = checkpoint['epoch']
    logger.info("Checkpoint {} (epoch {}) loaded.".format(checkpoint_file, start_epoch))
else:
    raise AttributeError('No checkpoint.')

total_param = []
for i in range(num_classes):
    for p in models[i].parameters():
        total_param.append(p.detach().cpu().numpy().flatten())

total_param = np.concatenate(total_param, axis=0)
print(total_param.shape)
total_param_abs = np.abs(total_param)

logger.info('Max (Abs): {}, Mean (Abs): {}, Std (Abs): {}, Max: {}, Min: {}, Mean: {}, Std: {}, # Abs < 0.001: {}'.format(
    total_param_abs.max(), total_param_abs.mean(), total_param_abs.std(ddof = 1),
    total_param.max(), total_param.min(), total_param.mean(), total_param.std(ddof = 1),
    np.sum(np.where(total_param_abs < 1e-3, 1, 0))
))
