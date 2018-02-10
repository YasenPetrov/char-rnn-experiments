from torch import optim

from log import get_logger

logger = get_logger(__name__)


# A map from string keys to PyTorch optimizer objects
OPTIM_MAP = {
    'sgd': optim.SGD,
    'adam': optim.Adam,
    'rmsprop': optim.RMSprop,
    'rprop': optim.Rprop,
    'adamax': optim.Adamax,
    'adagrad': optim.Adagrad,
    'adadelta': optim.Adadelta
}


def get_optimizer(optimizer_spec, model):
    """
    Return a PyTorch optimizer given a dictionary of specs for it
    :param optimizer_spec(dict): A dictionary containing a 'name' key, the value for which should be one of the keys in
    OPTIM_MAP and a 'kwargs' key, the value for which should be a dict with parameters to be passes to the constructor
    of the optimizer object
    :param model(torch.nn.Model): The PyTorch model the parameters of which the requester optimizer will operate on
    :return(torch.optim.Optimizer): The requested optimizer object
    """
    optim_name = optimizer_spec['name'].lower()
    if optim_name == '' or optim_name is None:
        logger.info('No optimizer name was supplied. An SGD optimizer will be used')
        return optim.SGD(model.parameters(), **optimizer_spec['kwargs'])
    elif optim_name not in OPTIM_MAP:
        raise ValueError ('Invalid optimizer name {0}. Available options are {1}'.format(
            optim_name,str(list(OPTIM_MAP.keys()))))
    else:
        return OPTIM_MAP[optim_name](model.parameters(), **optimizer_spec['kwargs'])