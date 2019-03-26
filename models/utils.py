import os
import json
from time import time
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

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
                 optimizer, learning_rate, num_epochs, l2_penalty, recurrent_dropout, linear_dropout,
                 batches_between_stats, max_grad_l2_norm=float('inf')):
        self.batches_between_stats = batches_between_stats
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
        self.max_grad_l2_norm=max_grad_l2_norm

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



def compute_gradient_rms(model, train_data, bptt, batch_size, use_gpu=False, max_batches=np.inf,
                         use_trained_hidden=False, best_init_hidden=None, loss_function=nn.CrossEntropyLoss()):
    # Initialize Mean Squared gradient norm accumulators
    for param in model.parameters():
        param.MS = 0 * param.data

    # Count how many batches were processed so we can average the MSG
    n_batches = 0

    start = time()
    for inputs, targets in train_data.get_batch_iterator(batch_size=batch_size, num_timesteps=bptt):
        if n_batches % max(int(max_batches / 20), 1) == 0:
            logger.info(f'Computing RMS gradient -- batch {n_batches}|{max_batches} in {time() - start :2f} seconds')
        if n_batches >= max_batches:
            break

        # Package inputs and targets into Variables
        if use_gpu:
            inputs = Variable(torch.Tensor(inputs)).cuda()
            targets = Variable(torch.LongTensor(targets)).cuda()
        else:
            inputs = Variable(torch.Tensor(inputs))
            targets = Variable(torch.LongTensor(targets))

        # Repackage hidden state
        if use_trained_hidden:
            init_hid = best_init_hidden
        else:
            init_hid = model.init_hidden(batch_size=batch_size)
        if use_gpu:
            if model.network_type == 'lstm':
                init_hid = tuple(v.cuda() for v in init_hid)
            else:
                init_hid = init_hid.cuda()
        model.hidden = init_hid

        model.zero_grad()

        logits = model(inputs)

        loss = loss_function(logits.contiguous().view(-1, logits.data.shape[-1]), targets.contiguous().view(-1))

        loss.backward()

        for param in model.parameters():
            if param.requires_grad:
                param.MS += param.grad.data ** 2

        n_batches += 1

    for param in model.parameters():
        param.RMS = torch.sqrt(param.MS)
        del param.MS