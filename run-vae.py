import os
from torch.utils.data import dataloader
import yaml
import torch
import argparse
import logging
import numpy as np
from tqdm import tqdm
from utils.logger import ColoredLogger
from utils.builder import dataloader_builder, vae_builder, optimizer_builder, lr_scheduler_builder
from utils.dataset import get_dataset_size
import torchvision.utils as tuitls


logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger(__name__)


# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--mode', '-m', default = 'train', help = 'the running mode, "train" or "inference"', type = str)
parser.add_argument('--cfg', '-c', default = os.path.join('configs', 'VAE.yaml'), help = 'Config File', type = str)
FLAGS = parser.parse_args()
CFG_FILE = FLAGS.cfg
MODE = FLAGS.mode

if MODE not in ['train', 'inference']:
    raise AttributeError('mode should be either "train" or "inference".')

with open(CFG_FILE, 'r') as cfg_file:
    cfg_dict = yaml.load(cfg_file, Loader=yaml.FullLoader)
    
model_params = cfg_dict.get('model', {})
dataset_params = cfg_dict.get('dataset', {})
optimizer_params = cfg_dict.get('optimizer', {})
lr_scheduler_params = cfg_dict.get('lr_scheduler', {})
trainer_params = cfg_dict.get('trainer', {})
inferencer_params = cfg_dict.get('inferencer', {})
stats_params = cfg_dict.get('stats', {})

logger.info('Building Models ...')
model = vae_builder(model_params)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

logger.info('Building dataloaders ...')
train_dataloader = dataloader_builder(dataset_params, split = 'train')
test_dataloader = dataloader_builder(dataset_params, split = 'test')
extra_dataloader = dataloader_builder(dataset_params, split = 'extra')

logger.info('Building optimizer and learning rate scheduler ...')
optimizer = optimizer_builder(model, optimizer_params)

lr_scheduler = lr_scheduler_builder(optimizer, lr_scheduler_params)

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
    if lr_scheduler is not None:
        lr_scheduler.last_epoch = start_epoch - 1
    logger.info("Checkpoint {} (epoch {}) loaded.".format(checkpoint_file, start_epoch))
elif MODE == "inference":
    raise AttributeError('There should be a checkpoint file for inference.')

batch_size = dataset_params.get('batch_size', 64)
total_train_samples = get_dataset_size(dataset_params.get('path', 'data'), 'train')
total_test_samples = get_dataset_size(dataset_params.get('path', 'data'), 'test')
total_extra_samples = get_dataset_size(dataset_params.get('path', 'data'), 'extra')


def train_one_epoch(epoch, extra = False):
    logger.info('Start training process in epoch {}.'.format(epoch + 1))
    model.train()
    losses = []
    if extra:
        dataloader = extra_dataloader
    else:
        dataloader = train_dataloader
    with tqdm(dataloader) as pbar:
        for data in pbar:
            optimizer.zero_grad()
            x, labels = data
            x = x.to(device)
            labels = labels.to(device)
            res = model(x, labels = labels)
            loss_dict = model.loss(
                *res,
                kl_weight = batch_size / total_train_samples,
                batch_size = batch_size,
                dataset_size = total_train_samples
            )
            loss = loss_dict['loss']
            loss.backward()
            optimizer.step()
            pbar.set_description('Epoch {}, loss: {:.8f}'.format(epoch + 1, loss.item()))
            losses.append(loss)
    mean_loss = torch.stack(losses).mean()
    logger.info('Finish training process in epoch {}, mean training loss: {:.8f}.'.format(epoch + 1, mean_loss))


def test_one_epoch(epoch):
    logger.info('Start evaluation process in epoch {}.'.format(epoch + 1))
    model.eval()
    losses = []
    with tqdm(test_dataloader) as pbar:
        for data in pbar:
            x, labels = data
            x = x.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                res = model(x, labels = labels)
                loss_dict = model.loss(
                    *res,
                    kl_weight = batch_size / total_test_samples,
                    batch_size = batch_size,
                    dataset_size = total_test_samples
                )
                loss = loss_dict['loss']
            pbar.set_description('Eval epoch {}, loss: {:.8f}'.format(epoch + 1, loss.item()))
            losses.append(loss)
    mean_loss = torch.stack(losses).mean()
    logger.info('Finish evaluation process in epoch {}, mean evaluation loss: {:.8f}'.format(epoch + 1, mean_loss))
    return mean_loss


def inference(epoch = -1):
    suffix = ""
    if 0 <= epoch < max_epoch:
        logger.info('Begin inference on checkpoint of epoch {} ...'.format(epoch + 1))
        suffix = "epoch_{}".format(epoch)
    else:
        logger.info('Begin inference ...')
    x, labels = next(iter(test_dataloader))
    x = x.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        recon = model.reconstruct(x, labels = labels)
    nrow = int(np.ceil(np.sqrt(batch_size)))
    reconstructed_dir = os.path.join(stats_dir, 'reconstructed_images')
    generated_dir = os.path.join(stats_dir, 'generated_images')
    if os.path.exists(reconstructed_dir) == False:
        os.makedirs(reconstructed_dir)
    if os.path.exists(generated_dir) == False:
        os.makedirs(generated_dir)
    tuitls.save_image(
        x.data,
        os.path.join(reconstructed_dir, "original_{}.png".format(suffix)),
        normalize = True,
        nrow = nrow
    )
    tuitls.save_image(
        recon.data,
        os.path.join(reconstructed_dir, "reconstructed_{}.png".format(suffix)),
        normalize = True,
        nrow = nrow
    )    
    sample_num = inferencer_params.get('sample_num', 144)
    nrow = int(np.ceil(np.sqrt(sample_num)))
    with torch.no_grad():
        samples = model.sample(sample_num, device, labels = labels)
    tuitls.save_image(
        samples.data,
        os.path.join(generated_dir, "generated_{}.png".format(suffix)),
        normalize = True,
        nrow = nrow
    )
    logger.info('Finish inference successfully.')
    

def train(start_epoch):
    global cur_epoch
    for epoch in range(start_epoch, max_epoch):
        cur_epoch = epoch
        logger.info('--> Epoch {}/{}'.format(epoch + 1, max_epoch))
        train_one_epoch(epoch)
        if trainer_params.get('extra', False):
            train_one_epoch(epoch, extra = True)
        loss = test_one_epoch(epoch)
        if lr_scheduler is not None:
            lr_scheduler.step()
        save_dict = {
            'epoch': epoch + 1, 
            'model_state_dict': model.state_dict(),
        }
        torch.save(save_dict, os.path.join(stats_dir, 'checkpoint.tar'))
        inference(epoch)


if __name__ == '__main__':
    if MODE == "train":
        train(start_epoch)
    else:
        inference()