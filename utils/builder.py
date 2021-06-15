def model_builder(model_params):
    from models.logistic_regression import CategoricalLogisticRegression
    model_name = model_params['name']
    if model_name == 'LogisticRegression':
        model = CategoricalLogisticRegression(**model_params)
    else:
        raise NotImplementedError('Invalid model name.')
    return model


def dataloader_builder(dataset_params, split = 'train'):
    from .dataset import SVHN_Dataset
    from torch.utils.data import DataLoader
    if split not in ['train', 'test', 'extra']:
        raise NotImplementedError('Invalid split name.')
    dataset = SVHN_Dataset(
        dataset_params.get('path', 'data'), 
        split = split, 
        **dataset_params,
    )
    return DataLoader(
        dataset,
        batch_size = dataset_params.get('batch_size', 64),
        shuffle = dataset_params.get('shuffle', True),
        num_workers = dataset_params.get('num_workers', 4),
        drop_last = dataset_params.get('drop_last', False)
    )


def optimizer_builder(model, optimizer_params):
    from torch.optim import SGD, ASGD, Adagrad, Adamax, Adadelta, Adam, AdamW, RMSprop
    type = optimizer_params.get('type', 'AdamW')
    params = optimizer_params.get('params', {})
    if type == 'SGD':
        optimizer = SGD(model.parameters(), **params)
    elif type == 'ASGD':
        optimizer = ASGD(model.parameters(), **params)
    elif type == 'Adagrad':
        optimizer = Adagrad(model.parameters(), **params)
    elif type == 'Adamax':
        optimizer = Adamax(model.parameters(), **params)
    elif type == 'Adadelta':
        optimizer = Adadelta(model.parameters(), **params)
    elif type == 'Adam':
        optimizer = Adam(model.parameters(), **params)
    elif type == 'AdamW':
        optimizer = AdamW(model.parameters(), **params)
    elif type == 'RMSprop':
        optimizer = RMSprop(model.parameters(), **params)
    else:
        raise NotImplementedError('Invalid optimizer type.')
    return optimizer


def lr_scheduler_builder(optimizer, lr_scheduler_params):
    from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR, CyclicLR, CosineAnnealingLR, LambdaLR, StepLR
    type = lr_scheduler_params.get('type', '')
    params = lr_scheduler_params.get('params', {})
    if type == 'MultiStepLR':
        scheduler = MultiStepLR(optimizer, **params)
    elif type == 'ExponentialLR':
        scheduler = ExponentialLR(optimizer, **params)
    elif type == 'CyclicLR':
        scheduler = CyclicLR(optimizer, **params)
    elif type == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, **params)
    elif type == 'LambdaLR':
        scheduler = LambdaLR(optimizer, **params)
    elif type == 'StepLR':
        scheduler = StepLR(optimizer, **params)
    elif type == '':
        scheduler = None
    else:
        raise NotImplementedError('Invalid learning rate scheduler type.')
    return scheduler


def vae_builder(model_params):
    from models.VAE.VAE import VAE
    from models.VAE.BetaVAE import BetaVAE
    from models.VAE.DisentangledBetaVAE import DisentangledBetaVAE
    from models.VAE.DFCVAE import DFCVAE
    from models.VAE.MSSIMVAE import MSSIMVAE
    model_name = model_params['name']
    if model_name == 'VAE':
        model = VAE(**model_params)
    elif model_name == 'BetaVAE':
        model = BetaVAE(**model_params)
    elif model_name == 'DisentangledBetaVAE':
        model = DisentangledBetaVAE(**model_params)
    elif model_name == 'DFCVAE':
        model = DFCVAE(**model_params)
    elif model_name == 'MSSIMVAE':
        model = MSSIMVAE(**model_params)
    else:
        raise NotImplementedError('Invalid model name.')
    return model