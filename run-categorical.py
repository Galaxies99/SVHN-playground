import os
import yaml
import torch
import argparse
import logging
from tqdm import tqdm
from utils.logger import ColoredLogger
from utils.builder import optimizer_builder, dataloader_builder, categorical_model_builder, lr_scheduler_builder
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
num_classes = trainer_params.get('num_classes', 10)
models = []
for _ in range(num_classes):
    models.append(categorical_model_builder(model_params))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
for i in range(num_classes):
    models[i].to(device)

logger.info('Building dataloaders ...')
train_dataloader = dataloader_builder(dataset_params, split = 'train')
test_dataloader = dataloader_builder(dataset_params, split = 'test')
extra_dataloader = dataloader_builder(dataset_params, split = 'extra')

logger.info('Building optimizer ...')
optimizers = []
for i in range(num_classes):
    optimizers.append(optimizer_builder(models[i], optimizer_params))
lr_schedulers = []
for i in range(num_classes):
    lr_schedulers.append(lr_scheduler_builder(optimizers[i], lr_scheduler_params))

logger.info('Checking checkpoints ...')
start_epoch = 0
max_epoch = trainer_params.get('max_epoch', 50)
stats_dir = os.path.join(stats_params.get('stats_dir', 'stats'), stats_params.get('stats_folder', 'temp'))
if os.path.exists(stats_dir) == False:
    os.makedirs(stats_dir)
checkpoint_file = os.path.join(stats_dir, 'checkpoint.tar')
if os.path.isfile(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    for i in range(num_classes):
        models[i].load_state_dict(checkpoint['model_state_dict'][i])
    start_epoch = checkpoint['epoch']
    for i in range(num_classes):
        if lr_schedulers[i] is not None:
            lr_schedulers[i].last_epoch = start_epoch - 1
    logger.info("Checkpoint {} (epoch {}) loaded.".format(checkpoint_file, start_epoch))


def train_one_epoch(epoch, extra = False):
    logger.info('Start training process in epoch {}.'.format(epoch + 1))
    for idx in range(num_classes):
        models[idx].train()
    losses = []
    if extra:
        dataloader = extra_dataloader
    else:
        dataloader = train_dataloader
    acc = 0
    cnt_all = 0
    with tqdm(dataloader) as pbar:
        for data in pbar:
            x, labels = data
            x = x.to(device)
            labels = labels.to(device)
            categorical_losses = []
            categorical_res = []
            for idx in range(num_classes):
                optimizers[idx].zero_grad()
                labels_idx = torch.where(labels == idx, 1, 0).to(device)
                res_idx = models[idx](x)
                categorical_res.append(res_idx.view(-1, 1))
                loss_idx = models[idx].loss(res_idx, labels_idx.view(-1), balance = num_classes)
                categorical_losses.append(loss_idx)
                loss_idx.backward()
                optimizers[idx].step()
            loss = torch.stack(categorical_losses, dim = 0).mean()
            res = torch.cat(categorical_res, dim = 1)
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
    logger.info('Finish training process in epoch {}, mean training loss: {:.8f}, mean accuracy: {:.6f}.'.format(epoch + 1, mean_loss, acc))


def test_one_epoch(epoch):
    logger.info('Start testing process in epoch {}.'.format(epoch + 1))
    for idx in range(num_classes):
        models[idx].eval()
    losses = []
    acc = 0
    cnt_all = 0
    with tqdm(test_dataloader) as pbar:
        for data in pbar:
            x, labels = data
            x = x.to(device)
            labels = labels.to(device)
            categorical_losses = []
            categorical_res = []
            for idx in range(num_classes):
                with torch.no_grad():
                    res_idx = models[idx](x)
                    categorical_res.append(res_idx.view(-1, 1))
                    labels_idx = torch.where(labels == idx, 1, 0).to(device)
                    loss = models[idx].loss(res_idx, labels_idx.view(-1), balance = num_classes)
                    categorical_losses.append(loss)
            loss = torch.stack(categorical_losses, dim = 0).mean()
            res = torch.cat(categorical_res, dim = 1)
            res_final = torch.argmax(res, dim = 1)
            cnt_all += len(labels)
            cur_acc = 0
            for i, label_sample in enumerate(labels):
                if int(res_final[i].item()) == int(label_sample.item()):
                    cur_acc += 1
            acc += cur_acc
            pbar.set_description('Epoch {}, loss: {:.8f}, accuracy: {:.6f}'.format(epoch + 1, loss.mean().item(), cur_acc / len(labels)))
            losses.append(loss)
    mean_loss = torch.stack(losses).mean()
    acc = acc / cnt_all
    logger.info('Finish testing process in epoch {}, mean testing loss: {:.8f}, accuracy: {:.6f}.'.format(epoch + 1, mean_loss, acc))
    return mean_loss, acc


def train(start_epoch):
    _, max_acc = test_one_epoch(-1)
    for epoch in range(start_epoch, max_epoch):
        logger.info('--> Epoch {}/{}'.format(epoch + 1, max_epoch))
        train_one_epoch(epoch)
        if trainer_params.get('extra', False):
            train_one_epoch(epoch, extra = True)
        _, acc = test_one_epoch(epoch)
        for idx in range(num_classes):
            if lr_schedulers[idx] is not None:
                lr_schedulers[idx].step()
        if acc > max_acc:
            max_acc = acc
            save_dict = {
                'epoch': epoch + 1,
                'model_state_dict': [model.state_dict() for model in models]
            }
            torch.save(save_dict, os.path.join(stats_dir, 'checkpoint.tar'))


if __name__ == '__main__':
    train(start_epoch = start_epoch)