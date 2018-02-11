import os
import json
import argparse
import time

import torch
from torch import nn

from config import EXPERIMENT_DIR, DEFAULT_MEMORY_LIMIT_BYTES
from datautils.dataset import Alphabet, Dataset
from models.rnn import get_rnn_for_hyperparams
from models.utils import RNN_Hyperparameters
from training import evaluate_rnn
from log import get_logger

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

    data = Dataset(args.test_text_file, alphabet, DEFAULT_MEMORY_LIMIT_BYTES)

    # TODO: work out timesteps
    chars_processed, loss = evaluate_rnn(model, data,
                        loss_function=nn.modules.loss.CrossEntropyLoss(),
                        num_timesteps=10000,
                        use_gpu=args.use_gpu)

    return chars_processed, loss


if __name__ == '__main__':
    # TODO: Help messages
    parser = argparse.ArgumentParser(description='''Runs an evaluation session''')

    parser.add_argument('experiment_name', metavar='experiment_name', type=str, help='The name of the experiment' + \
                        'The best model from that expriment will be used for evaluation.')
    parser.add_argument('test_text_file', metavar='test_text_file', type=str,
                        help='Path to a UTF-8 encoded file for the model to be evaluated against')
    parser.add_argument('--gpu', dest='use_gpu', action='store_const', default=False, const=True,
                        help='If set, evaluation is performed on a GPU, if available')
    parser.add_argument('--gpu_id', default='0', type=str, help='id for CUDA_VISIBLE_DEVICES')

    args = parser.parse_args()
    if args.use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    eval_start = time.time()
    chars_processed, loss = evaluate(args)
    eval_end = time.time()
    seconds_elapsed = eval_end - eval_start
    m, s = divmod(int(seconds_elapsed), 60)
    h, m = divmod(m, 60)

    logger.info('Finished evaluation in {h:02d}:{m:02d}:{s:02d}'.format(h=h, m=m, s=s))
    logger.info('Processed {0} chars, ~{1:.2f} per sec, Resulting BPC: {2:.5f}'.format(chars_processed,
                                                                                       chars_processed / seconds_elapsed,
                                                                                       loss))