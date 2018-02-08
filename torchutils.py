from torch import optim

from log import get_logger

logger = get_logger(__name__)

def get_optimizer(optimizer_spec, model):
    """

    :param optimizer_spec:
    :param model:
    :return:
    """
    optim_name = optimizer_spec['name']
    if optim_name == '' or optim_name is None:
        logger.info('No optimizer name was supplied. An SGD optimizer will be used')
        return optim.SGD(model.parameters(), **optimizer_spec['kwargs'])
    elif optim_name.lower() == 'sgd':
        return optim.SGD(model.parameters(), **optimizer_spec['kwargs'])