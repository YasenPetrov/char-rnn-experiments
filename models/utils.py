import os
import json
from collections import defaultdict

from log import get_logger

logger = get_logger(__name__)

class Hyperparameters:
    def dump_to_json(self, filename):
        try:
            if os.path.exists(filename):
                os.remove(filename)
                logger.warning('Overwriting hyperparam dict in {0}'.format(filename))
            with open(filename, 'w+') as fp:
                json.dump(self.__dict__, fp, indent=2)
        except PermissionError:
            logger.warning('Cannot delete or write to {0}'.format(filename))

    def to_string(self):
        return json.dumps(self.__dict__, indent=2)


class RNN_Hyperparameters(Hyperparameters):
    def __init__(self, network_type, hidden_size, num_layers, batch_size, num_timesteps, reset_state_every,
                 optimizer, learning_rate, num_epochs, l2_penalty, recurrent_dropout, linear_dropout):
        self.linear_dropout = linear_dropout
        self.recurrent_dropout = recurrent_dropout
        self.l2_penalty = l2_penalty
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.reset_state_every = reset_state_every
        self.num_timesteps = num_timesteps
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.network_type = network_type

    @staticmethod
    def from_json(filename):
        with open(filename, 'r') as fp:
            param_dict = json.load(fp)
        return RNN_Hyperparameters(**param_dict)


class Ngram_Hyperparameters(Hyperparameters):
    def __init__(self, n, delta, adapt):
        self.adapt = adapt
        self.delta = delta
        self.n = n

    @staticmethod
    def from_json(filename):
        with open(filename, 'r') as fp:
            param_dict = json.load(fp)
        return Ngram_Hyperparameters(**param_dict)