def model_builder(model_params):
    from models.logistic_regression import CategoricalLogisticRegression
    model_name = model_params['name']
    if model_name == 'LogisticRegression':
        model = CategoricalLogisticRegression(**model_params)
    else:
        raise NotImplementedError('Invalid model name.')
    return model


def dataloader_builder(dataset_params, split = 'train'):
    from .dataset import get_dataset
    from torch.utils.data import DataLoader
    if split not in ['train', 'test', 'extra']:
        raise NotImplementedError('Invalid split name.')
    dataset = get_dataset(
        dataset_params.get('path', 'data'), 
        split = split, 
        gray_scale = dataset_params.get('gray_scale', False),
        hog_feature = dataset_params.get('hog_feature', False)
    )
    return DataLoader(
        dataset,
        batch_size = dataset_params.get('batch_size', 64),
        shuffle = dataset_params.get('shuffle', True),
        num_workers = dataset_params.get('num_workers', 4)
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