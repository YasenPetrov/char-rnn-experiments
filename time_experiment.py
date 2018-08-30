import argparse
import os
import json
import shutil
import platform
import time

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import pandas as pd
from models.rnn import RNN_LM

from utils.general_utils import dict_of_lists_to_list_of_dicts
from log import get_logger
from config import TIME_EXPERIMENT_DIR, EXPERIMENT_SPEC_FILENAME

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
        _ = sm(out).cpu()
        # Synchronize - crucial if we want to benchmark GPU operations
        if use_gpu:
            torch.cuda.synchronize()
        end = time.perf_counter()

        # Ignore first iteration - it always takes a while longer on a GPU and will skew our results
        if i > 0:
            times[i - 1] = end - start
    return times.mean(), times.std()


def perform_time_experiment(cmd_args):
    logger.info('\n{0}\nStarting timing experiment {1}\n{2}'.format('=' * 30, cmd_args.experiment_name, '=' * 30))
    # We expect to find a spec.json file in a folder with the same name as the experiment under the main experiment dir
    experiment_folder = os.path.join(TIME_EXPERIMENT_DIR, cmd_args.experiment_name)

    if not os.path.exists(experiment_folder):
        raise FileNotFoundError(experiment_folder)

    spec_filepath = os.path.join(experiment_folder, EXPERIMENT_SPEC_FILENAME)

    if not os.path.exists(spec_filepath):
        raise FileNotFoundError(spec_filepath)

    with open(spec_filepath, 'r') as fp:
        spec = json.load(fp)

    start_time = time.time()
    # Store the mean prediction time and the SD on that for each parameter combination
    means, sds = [], []
    param_list = dict_of_lists_to_list_of_dicts(spec)
    res_df = pd.DataFrame(param_list)
    perc = 10
    for i, p in enumerate(param_list):
        # Log at every 10% of the models tested
        if 100 * i / len(param_list) > perc:
            logger.info('Tested {0} out of {1} models'.format(i, len(param_list)))
            perc += 10
        try:
            mu, s = measure_prediction_time(**p)
            means.append(mu)
            sds.append(s)
        except Exception:
            logger.exception('While trying to time for hyperparams {0}'.format(p))
    res_df.loc[:, 'mean_time'] = means
    res_df.loc[:, 'time_sd'] = sds

    end_time = time.time()
    time_elapsed_sec = end_time - start_time

    # Build file structure
    experiment_out_dir = os.path.join(experiment_folder, 'out')
    if os.path.exists(experiment_out_dir):
        logger.info('Removing old results directory {0}'.format(experiment_out_dir))
        shutil.rmtree(experiment_out_dir)
    os.makedirs(experiment_out_dir)

    # Store the experiment results in a csv
    res_df.to_csv(os.path.join(experiment_out_dir, 'results.csv'), index=False)

    # Store system info so I know which machine I've ran this from
    system_dict = {'cwd': os.getcwd(),
                   'gpu': torch.cuda.get_device_name(int(cmd_args.gpu_id)),
                   'cpu': platform.processor()}
    system_dict_filename = os.path.join(experiment_out_dir, 'system.json')
    with open(system_dict_filename, 'w') as fp:
        json.dump(system_dict, fp)

    # Log the time elapsed
    with open(os.path.join(experiment_out_dir, 'time_sec.txt'), 'w') as fp:
        fp.write(str(time_elapsed_sec))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Runs a training or evaluation session''')

    parser.add_argument('experiment_name', metavar='experiment_name', type=str, help='The name of the experiment')
    parser.add_argument('--empty_cache', dest='empty_cache', action='store_const', default=False, const=True,
                        help='If set, the GPU cache is emptied before each timing')
    parser.add_argument('--gpu_id', default='0', type=str, help='id for CUDA_VISIBLE_DEVICES')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    perform_time_experiment(args)
