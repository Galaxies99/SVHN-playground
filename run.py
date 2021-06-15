import os
import yaml
import torch
import argparse
import logging
from tqdm import tqdm
from utils.logger import ColoredLogger
from utils.builder import optimizer_builder, dataloader_builder, model_builder, lr_scheduler_builder
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
dataset_params = cfg_dict.get('dataset', {})
optimizer_params = cfg_dict.get('optimizer', {})
lr_scheduler_params = cfg_dict.get('lr_scheduler', {})
trainer_params = cfg_dict.get('trainer', {})
stats_params = cfg_dict.get('stats', {})

logger.info('Building models ...')
model = model_builder(model_params)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

logger.info('Building dataloaders ...')
train_dataloader = dataloader_builder(dataset_params, split = 'train')
test_dataloader = dataloader_builder(dataset_params, split = 'test')
extra_dataloader = dataloader_builder(dataset_params, split = 'extra')

logger.info('Building optimizer ...')
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


def train_one_epoch(epoch, extra = False):
    logger.info('Start training process in epoch {}.'.format(epoch + 1))
    model.train()
    losses = []
    if extra:
        dataloader = extra_dataloader
    else:
        dataloader = train_dataloader
    acc = 0
    cnt_all = 0
    with tqdm(dataloader) as pbar:
        for data in pbar:
            optimizer.zero_grad()
            x, labels = data
            x = x.to(device)
            labels = labels.to(device)
            res = model(x)
            loss = model.loss(res, labels.view(-1))
            loss.backward()
            optimizer.step()
            res_final = torch.argmax(res, dim = 1)
            cnt_all += len(labels)
            cur_acc = 0
            for i, label_sample in enumerate(labels):
                if int(res_final[i].item()) == int(label_sample.item()):
                    cur_acc += 1
            acc += cur_acc
            pbar.set_description('Epoch {}, loss: {:.8f}, accuracy: {:.6f}'.format(epoch + 1, loss.mean().item(), cur_acc / len(labels)))
            losses.append(loss.mean())
    mean_loss = torch.stack(losses).mean()
    acc = acc / cnt_all
    logger.info('Finish training process in epoch {}, mean training loss: {:.8f}, mean accuracy: {:.6f}'.format(epoch + 1, mean_loss, acc))


def test_one_epoch(epoch):
    logger.info('Start testing process in epoch {}.'.format(epoch + 1))
    model.eval()
    losses = []
    acc = 0
    cnt_all = 0
    with tqdm(test_dataloader) as pbar:
        for data in pbar:
            x, labels = data
            x = x.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                res = model(x)
                loss = model.loss(res, labels.view(-1))
            res_final = torch.argmax(res, dim = 1)
            cnt_all += len(labels)
            cur_acc = 0
            for i, label_sample in enumerate(labels):
                if int(res_final[i].item()) == int(label_sample.item()):
                    cur_acc += 1
            acc += cur_acc
            pbar.set_description('Epoch {}, loss: {:.8f}, accuracy: {:.6f}'.format(epoch + 1, loss.mean().item(), cur_acc / len(labels)))
            losses.append(loss.mean())
    mean_loss = torch.stack(losses).mean()
    acc = acc / cnt_all
    logger.info('Finish testing process in epoch {}, mean testing loss: {:.8f}, mean accuracy: {:.6f}.'.format(epoch + 1, mean_loss, acc))
    return mean_loss, acc


def train(start_epoch):
    _, max_acc = test_one_epoch(start_epoch - 1)
    max_acc_epoch = 0
    for epoch in range(start_epoch, max_epoch):
        logger.info('--> Epoch {}/{}'.format(epoch + 1, max_epoch))
        train_one_epoch(epoch)
        if trainer_params.get('extra', False):
            train_one_epoch(epoch, extra = True)
        _, acc = test_one_epoch(epoch)
        if lr_scheduler is not None:
            lr_scheduler.step()
        if acc > max_acc:
            max_acc = acc
            max_acc_epoch = epoch + 1
            save_dict = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
            }
            torch.save(save_dict, os.path.join(stats_dir, 'checkpoint.tar'))
    logger.info('Training Finished. Max accuracy: {:.6f}, in epoch {}'.format(max_acc, max_acc_epoch))


if __name__ == '__main__':
    train(start_epoch = start_epoch)