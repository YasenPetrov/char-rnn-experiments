import time

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from models.rnn import RNN_LM
from log import get_logger

logger = get_logger(__name__)

def measure_prediction_time(hidden_dim, alphabet_size, seq_length, batch_size, network_type, num_layers, use_gpu,
                            num_trials=10, empty_cache=False):
    """
    Estimates the prediction time for a network with the given hyperparameters. Runs several predictions and reports
    the average time in seconds and the standard deviation
    :param hidden_dim: See models.rnn.RNN_LM
    :param alphabet_size: See models.rnn.RNN_LM
    :param seq_length: See models.rnn.RNN_LM
    :param batch_size: See models.rnn.RNN_LM
    :param network_type: See models.rnn.RNN_LM
    :param num_layers: See models.rnn.RNN_LM
    :param use_gpu: See models.rnn.RNN_LM
    :param num_trials: The number of predictions the estimate is calculated over
    :return(float, float): Mean prediction time, standard deviation
    """
    if empty_cache:
        try:
            torch.cuda.empty_cache()
        except Exception:
            logger.exception('Cannot empty cache')

    model = RNN_LM(
        hidden_dim=hidden_dim,
        alphabet_size=alphabet_size,
        batch_size=batch_size,
        num_layers=num_layers,
        network_type=network_type,
        use_gpu=use_gpu
    )

    if use_gpu:
        model.cuda()

    sm = nn.Softmax(dim=1)
    times = np.empty(num_trials)

    for i in range(num_trials + 1):
        start = time.perf_counter()

        # Initialize hiden state
        model.hidden = model.init_hidden(batch_size=batch_size)

        # Random inputs
        inputs = torch.Tensor(np.random.randint(0, alphabet_size, (batch_size, seq_length, alphabet_size)))
        if use_gpu:
            inputs = Variable(inputs, volatile=True).cuda()
        else:
            inputs = Variable(inputs, volatile=True)

        # Forward pass
        out = model(inputs).view(-1, alphabet_size)

        # Calculate probabilities and bring them back to the CPU (if the GPU was used)
        probas = sm(out).cpu()
        # Synchronize - crucial if we want to benchmark GPU operations
        if use_gpu:
            torch.cuda.synchronize()
        end = time.perf_counter()

        # Ignore first iteration - it always takes a while longer on a GPU and will skew our results
        if i > 0:
            times[i - 1] = end - start
    return times.mean(), times.std()