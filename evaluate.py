import os
import json
import argparse
import shutil

import torch
from torch import nn
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import EXPERIMENT_DIR, DEFAULT_MEMORY_LIMIT_BYTES
from datautils.dataset import Alphabet, Dataset
from models.rnn import get_rnn_for_hyperparams
from models.utils import RNN_Hyperparameters
from training import evaluate_rnn
from log import get_logger
from utils import dict_of_lists_to_list_of_dicts
from torchutils import get_optimizer

logger = get_logger(__name__)


def evaluate(args):
    # Load best model from experiment
    experiment_folder = os.path.join(EXPERIMENT_DIR, args.experiment_name)

    if not os.path.exists(experiment_folder):
        raise FileNotFoundError(experiment_folder)

    results_file = os.path.join(experiment_folder, 'out/results.json')
    alphabet_file = os.path.join(experiment_folder, 'alphabet.json')

    alphabet = Alphabet.from_json(alphabet_file)

    with open(results_file, 'r') as fp:
        exp_results = json.load(fp)

    best_key = exp_results['best_key']
    checkpoint_path = exp_results[best_key]['path']
    hyperparams = RNN_Hyperparameters(**exp_results[best_key]['config'])

    logger.info('Evaluating model with results:\n{hps}'.format(hps=json.dumps(exp_results[best_key], indent=2)))

    model = get_rnn_for_hyperparams(hyperparams, alphabet.get_size(), args.use_gpu)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])

    if args.use_gpu:
        model.cuda()

    # Load spec file
    eval_dir = os.path.join(experiment_folder, 'eval')
    spec_file = os.path.join(eval_dir, args.test_name + '.json')
    with open(spec_file, 'r') as fp:
        spec = json.load(fp)

    eval_out_dir = os.path.join(eval_dir, args.test_name)
    if os.path.exists(eval_out_dir):
        shutil.rmtree(eval_out_dir)

    # Make sure there is at most one static evaluation - hyperparams have no effect there
    if False in spec['dynamic']:
        spec['dynamic'] = [x for x in spec['dynamic'] if x]
        hypers_list = dict_of_lists_to_list_of_dicts(spec)
        hypers_list += dict_of_lists_to_list_of_dicts({'file': spec['file'],
                                                       'dynamic': [False],
                                                       'num_timesteps': spec['stats_interval'],
                                                       'stats_interval': spec['stats_interval'],
                                                       'learning_rate': [None],
                                                       'decay_coef': [None]})
    else:
        hypers_list = dict_of_lists_to_list_of_dicts(spec)

    loss_log = {}
    results = {}
    best_final_loss = None
    best_id = None

    for i, hypers in enumerate(hypers_list):
        data = Dataset(hypers['file'], alphabet, DEFAULT_MEMORY_LIMIT_BYTES)

        model.load_state_dict(checkpoint['state_dict'])
        if args.use_gpu:
            model.cuda()

        chars_processed, loss, losses = evaluate_rnn(model, data,
                                                    loss_function=nn.modules.loss.CrossEntropyLoss(),
                                                    num_timesteps=hypers['num_timesteps'],
                                                    use_gpu=args.use_gpu,
                                                    dynamic=hypers['dynamic'],
                                                    stats_interval=hypers['stats_interval'],
                                                    record_stats=True,
                                                    learning_rate=hypers['learning_rate'],
                                                    decay_coef=hypers['decay_coef'])

        if best_final_loss is None or loss < best_final_loss:
            best_final_loss = loss
            best_id = i

        hypers['final_loss'] = loss
        results[i] = hypers
        losses = pd.DataFrame(losses)
        loss_log[i] = losses

        out_dir = os.path.join(eval_out_dir, str(i))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        losses.to_csv(os.path.join(out_dir, 'stats.csv'), index=False)
        with open(os.path.join(out_dir, 'hyperparams.json'), 'w+') as fp:
            json.dump(hypers, fp, indent=2)

        # Plot losses and save image
        plt.figure(figsize=(12, 8))
        plt.grid()
        plt.plot(losses.chars_processed, losses.loss)
        plt.legend(fontsize=14)
        plt.xlabel('Chars processed', fontsize=14)
        plt.ylabel('BPC', fontsize=14)
        hypers_str = str(hypers)
        plt.title(hypers_str[:len(hypers_str) // 2] + '\n' + hypers_str[len(hypers_str) // 2:], fontsize=10)

        img_path = os.path.join(out_dir, 'losses.png')
        if os.path.exists(img_path):
            os.remove(img_path)
        plt.savefig(img_path)

    # Plot losses and save image
    plt.figure(figsize=(12, 8))
    plt.grid()
    for i, losses in loss_log.items():
        plt.plot(losses.chars_processed, losses.loss, label=str(i))
    plt.legend(fontsize=14)
    plt.xlabel('Chars processed', fontsize=14)
    plt.ylabel('BPC', fontsize=14)

    img_path = os.path.join(eval_out_dir, 'losses.png')
    if os.path.exists(img_path):
        os.remove(img_path)
    plt.savefig(img_path)

    with open(os.path.join(eval_out_dir, 'results.json'), 'w+') as fp:
        results['best_key'] = best_id
        json.dump(results, fp, indent=2)


if __name__ == '__main__':
    # TODO: Help messages
    parser = argparse.ArgumentParser(description='''Runs an evaluation session''')

    parser.add_argument('experiment_name', metavar='experiment_name', type=str, help='The name of the experiment' + \
                        'The best model from that expriment will be used for evaluation.')
    parser.add_argument('test_name', metavar='test_name', type=str,
                        help='Path to a UTF-8 encoded file for the model to be evaluated against')
    parser.add_argument('--gpu', dest='use_gpu', action='store_const', default=False, const=True,
                        help='If set, evaluation is performed on a GPU, if available')
    parser.add_argument('--gpu_id', default='0', type=str, help='id for CUDA_VISIBLE_DEVICES')

    args = parser.parse_args()
    if args.use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    evaluate(args)
    # eval_start = time.time()
    # chars_processed, loss = evaluate(args)
    # eval_end = time.time()
    # seconds_elapsed = eval_end - eval_start
    # m, s = divmod(int(seconds_elapsed), 60)
    # h, m = divmod(m, 60)
    #
    # logger.info('Finished evaluation in {h:02d}:{m:02d}:{s:02d}'.format(h=h, m=m, s=s))
    # logger.info('Processed {0} chars, ~{1:.2f} per sec, Resulting BPC: {2:.5f}'.format(chars_processed,
    #                                                                                    chars_processed / seconds_elapsed,
    #                                                                                    loss))