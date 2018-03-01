import argparse
import os
import json
import shutil
import platform
import time

import pandas as pd
import torch

from utils import dict_of_lists_to_list_of_dicts
from timing import measure_prediction_time
from log import get_logger
from config import TIME_EXPERIMENT_DIR, EXPERIMENT_SPEC_FILENAME

logger = get_logger(__name__)


def perform_time_experiment(args):
    logger.info('\n{0}\nStarting timing experiment {1}\n{2}'.format('=' * 30, args.experiment_name, '=' * 30))
    # We expect to find a spec.json file in a folder with the same name as the experiment under the main experiment dir
    experiment_folder = os.path.join(TIME_EXPERIMENT_DIR, args.experiment_name)

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
                   'gpu': torch.cuda.get_device_name(int(args.gpu_id)),
                   'cpu': platform.processor()}
    system_dict_filename = os.path.join(experiment_out_dir, 'system.json')
    with open(system_dict_filename, 'w') as fp:
        json.dump(system_dict, fp)

    # Log the time elapsed
    with open(os.path.join(experiment_out_dir, 'time_sec.txt'), 'w') as fp:
        fp.write(str(time_elapsed_sec))


if __name__ == '__main__':
    # TODO: Help messages
    parser = argparse.ArgumentParser(description='''Runs a training or evaluation session''')

    parser.add_argument('experiment_name', metavar='experiment_name', type=str, help='The name of the experiment')
    parser.add_argument('--empty_cache', dest='empty_cache', action='store_const', default=False, const=True,
                        help='If set, the GPU cache is emptied before each timing')
    parser.add_argument('--gpu_id', default='0', type=str, help='id for CUDA_VISIBLE_DEVICES')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    perform_time_experiment(args)