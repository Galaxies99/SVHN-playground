import os
import cv2
import yaml
import torch
import argparse
import logging
import numpy as np
from tqdm import tqdm
from utils.logger import ColoredLogger
from utils.builder import optimizer_builder, dataloader_builder, model_builder, lr_scheduler_builder
import argparse
import torchvision.utils as tuitls
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', '-c', default = os.path.join('configs', 'VAE', 'default.yaml'), help = 'Config File', type = str)
FLAGS = parser.parse_args()
CFG_FILE = FLAGS.cfg

with open(CFG_FILE, 'r') as cfg_file:
    cfg_dict = yaml.load(cfg_file, Loader=yaml.FullLoader)


model_params = cfg_dict.get('model', {})
dataset_params = cfg_dict.get('dataset', {})
trainer_params = cfg_dict.get('trainer', {})
stats_params = cfg_dict.get('stats', {})

logger.info('Building models ...')
model = model_builder(model_params)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

logger.info('Reading Datasets ...')
test_dataloader = dataloader_builder(dataset_params, split = 'test')

logger.info('Checking checkpoints ...')
start_epoch = 0
max_epoch = trainer_params.get('max_epoch', 50)
stats_dir = os.path.join(stats_params.get('stats_dir', 'stats'), stats_params.get('stats_folder', 'temp'))
if os.path.exists(stats_dir) == False:
    os.makedirs(stats_dir)
checkpoint_file = os.path.join(stats_dir, 'checkpoint.tar')
if os.path.isfile(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    logger.info("Checkpoint {} (epoch {}) loaded.".format(checkpoint_file, start_epoch))
else:
    raise AttributeError('Checkpoint not found.')

target_layer = model.layer5[-1]
cam = GradCAM(model = model, target_layer = target_layer, use_cuda = True)

samples, _ = next(iter(test_dataloader))
samples = samples.to(device)
grayscale_cam = cam(input_tensor = samples)
input_rgb = samples.detach().cpu().numpy()
input_rgb = input_rgb.transpose(0, 2, 3, 1)
input_rgb = (input_rgb + 1) / 2

batch_size = samples.shape[0]
nrow = int(np.floor(np.sqrt(batch_size)))
images = []
for i in tqdm(range(nrow * nrow)):
    input_rgb_sample = input_rgb[i, :, :, :]
    cam_image = show_cam_on_image(input_rgb_sample, grayscale_cam[i, :], use_rgb = True)
    cam_image = cam_image.transpose(2, 0, 1)
    images.append(cam_image)

images = torch.FloatTensor(np.array(images))

tuitls.save_image(
    images.data,
    os.path.join(stats_dir, "gradcam.png"),
    normalize = True,
    nrow = nrow
)
