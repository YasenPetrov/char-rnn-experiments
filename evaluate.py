import os
import json
import argparse
import shutil
import time
import copy

import numpy as np
import torch
from torch import nn
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import EXPERIMENT_DIR, DEFAULT_MEMORY_LIMIT_BYTES, EXPERIMENT_SPEC_FILENAME, DATA_DIR
from datautils.dataset import Alphabet, Dataset
from models.rnn import get_rnn_for_hyperparams, get_step_rnn_for_rnn
from models.utils import RNN_Hyperparameters, compute_gradient_rms
from training import evaluate_rnn, evaluate_rnn_lhuc_sparse
from log import get_logger
from utils.general_utils import dict_of_lists_to_list_of_dicts
from utils.debugging import apply_hooks

logger = get_logger(__name__)


def evaluate_lhuc_or_sparse(args):
    # Load best model from experiment
    experiment_folder = os.path.join(EXPERIMENT_DIR, args.experiment_name)

    # We expect to find a spec.json file in a folder with the same name as the experiment under the main experiment dir
    experiment_folder = os.path.join(EXPERIMENT_DIR, args.experiment_name)
    spec_filepath = os.path.join(experiment_folder, EXPERIMENT_SPEC_FILENAME)
    with open(spec_filepath, 'r') as fp:
        spec = json.load(fp)
    # Check if data files exist
    data_folder = os.path.join(DATA_DIR, spec['data_folder'])
    train_filename = os.path.join(data_folder, 'train.txt')

    # Load eval spec file
    eval_dir = os.path.join(experiment_folder, 'eval')
    spec_file = os.path.join(eval_dir, args.test_name + '.json')
    with open(spec_file, 'r') as fp:
        full_spec = json.load(fp)

    # Which models are we testing on - those are specified by integer ids
    model_keys = full_spec['models']
    del full_spec['models']

    for model_key in model_keys:
        # TODO: This is hacky, figure out a neat way
        spec = copy.deepcopy(full_spec)

        if not os.path.exists(experiment_folder):
            raise FileNotFoundError(experiment_folder)

        results_file = os.path.join(experiment_folder, 'out/results.json')
        alphabet_file = os.path.join(experiment_folder, 'alphabet.json')

        alphabet = Alphabet.from_json(alphabet_file)

        # Get training data for RMS estimation
        train_data = Dataset(train_filename, alphabet)

        with open(results_file, 'r') as fp:
            exp_results = json.load(fp)

        # best_key = exp_results['best_key']
        checkpoint_path = exp_results[str(model_key)]['path']
        hyperparams = RNN_Hyperparameters(**exp_results[str(model_key)]['config'])
        checkpoint = torch.load(checkpoint_path)

        logger.info(
            'Evaluating model with results:\n{hps}'.format(hps=json.dumps(exp_results[str(model_key)], indent=2)))

        main_eval_out_dir = os.path.join(eval_dir, args.test_name)
        eval_out_dir = os.path.join(main_eval_out_dir, str(model_key))
        if os.path.exists(eval_out_dir):
            shutil.rmtree(eval_out_dir)

        # We can specify how much of the validation file to use -- if we haven't or we've spacified <0, use them all
        if "eval_char_count" not in spec or spec["eval_char_count"][0] < 0:
            spec["eval_char_count"] = [np.inf]

        hypers_list = dict_of_lists_to_list_of_dicts(spec)

        ###
        loss_log = {}
        results = {}
        best_final_loss = None
        best_id = None
        total_time = 0
        ###

        for i, hypers in enumerate(hypers_list):

            logger.info(f'Evaluating for {i+1}|{len(hypers_list)}:\n{json.dumps(hypers, indent=2)}')
            data = Dataset(hypers['file'], alphabet, DEFAULT_MEMORY_LIMIT_BYTES)

            model = get_rnn_for_hyperparams(hyperparams, alphabet.get_size(), args.use_gpu)
            model.load_state_dict(checkpoint['state_dict'])
            model = get_step_rnn_for_rnn(model, step_model_type=hypers['adapt_rule'])

            # Register debugging hooks to detect nan gradients
            if args.debug:
                apply_hooks(model)

            if args.use_gpu:
                model.cuda()

            start = time.time()
            chars_processed, loss, losses = evaluate_rnn_lhuc_sparse(model, data,
                                                                     loss_function=nn.modules.loss.CrossEntropyLoss(),
                                                                     num_timesteps=hypers['num_timesteps'],
                                                                     use_gpu=args.use_gpu,
                                                                     stats_interval=hypers['stats_interval'],
                                                                     record_stats=True,
                                                                     learning_rate=hypers['learning_rate'],
                                                                     num_chars_to_read=hypers["eval_char_count"],
                                                                     adapt_rule=hypers['adapt_rule'],
                                                                     use_in_recurrent=hypers['use_in_recurrent'])

            if best_final_loss is None or loss < best_final_loss:
                best_final_loss = loss
                best_id = i

            end = time.time()
            hypers['time_min'] = (end - start) / 60
            total_time += end - start

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

        # Save results for model ID
        with open(os.path.join(eval_out_dir, 'results.json'), 'w+') as fp:
            results['best_key'] = best_id
            results['total_time_min'] = total_time / 60
            json.dump(results, fp, indent=2)

def evaluate(args, save_images=False):
    # Load best model from experiment
    experiment_folder = os.path.join(EXPERIMENT_DIR, args.experiment_name)

    # We expect to find a spec.json file in a folder with the same name as the experiment under the main experiment dir
    experiment_folder = os.path.join(EXPERIMENT_DIR, args.experiment_name)
    spec_filepath = os.path.join(experiment_folder, EXPERIMENT_SPEC_FILENAME)
    with open(spec_filepath, 'r') as fp:
        spec = json.load(fp)
    # Check if data files exist
    data_folder = os.path.join(DATA_DIR, spec['data_folder'])
    train_filename = os.path.join(data_folder, 'train.txt')

    # Load eval spec file
    eval_dir = os.path.join(experiment_folder, 'eval')
    spec_file = os.path.join(eval_dir, args.test_name + '.json')
    with open(spec_file, 'r') as fp:
        full_spec = json.load(fp)

    # Which models are we testing on - those are specified by integer ids
    model_keys = full_spec['models']
    del full_spec['models']

    for model_key in model_keys:
        # TODO: This is hacky, figure out a neat way
        spec = copy.deepcopy(full_spec)

        if not os.path.exists(experiment_folder):
            raise FileNotFoundError(experiment_folder)

        results_file = os.path.join(experiment_folder, 'out/results.json')
        alphabet_file = os.path.join(experiment_folder, 'alphabet.json')

        alphabet = Alphabet.from_json(alphabet_file)

        # Get training data for RMS estimation
        train_data = Dataset(train_filename, alphabet)

        with open(results_file, 'r') as fp:
            exp_results = json.load(fp)

        # best_key = exp_results['best_key']
        checkpoint_path = exp_results[str(model_key)]['path']
        hyperparams = RNN_Hyperparameters(**exp_results[str(model_key)]['config'])

        logger.info('Evaluating model with results:\n{hps}'.format(hps=json.dumps(exp_results[str(model_key)], indent=2)))

        model = get_rnn_for_hyperparams(hyperparams, alphabet.get_size(), args.use_gpu)

        # Register debugging hooks to detect nan gradients
        if args.debug:
            apply_hooks(model)

        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])

        if args.use_gpu:
            model.cuda()

        main_eval_out_dir = os.path.join(eval_dir, args.test_name)
        eval_out_dir = os.path.join(main_eval_out_dir, str(model_key))
        if os.path.exists(eval_out_dir):
            shutil.rmtree(eval_out_dir)


        # We can specify how much of the validation file to use -- if we haven't or we've spacified <0, use them all
        if "eval_char_count" not in spec or spec["eval_char_count"][0] < 0:
            spec["eval_char_count"] = [np.inf]

        # Make sure there is at most one static evaluation - hyperparams have no effect there
        if False in spec['dynamic']:
            spec['dynamic'] = [x for x in spec['dynamic'] if x]
            hypers_list = dict_of_lists_to_list_of_dicts(spec)
            hypers_list += dict_of_lists_to_list_of_dicts({'file': spec['file'],
                                                           'dynamic': [False],
                                                           'num_timesteps': spec['stats_interval'],
                                                           'stats_interval': spec['stats_interval'],
                                                           'learning_rate': [None],
                                                           'decay_coef': [None],
                                                           'eval_char_count': spec['eval_char_count']})
        else:
            hypers_list = dict_of_lists_to_list_of_dicts(spec)

        loss_log = {}
        results = {}
        best_final_loss = None
        best_id = None

        total_time = 0

        rms_grad_stats = dict()
        for i, hypers in enumerate(hypers_list):

            logger.info(f'Evaluating for {i+1}|{len(hypers_list)}:\n{json.dumps(hypers, indent=2)}')
            data = Dataset(hypers['file'], alphabet, DEFAULT_MEMORY_LIMIT_BYTES)

            model.load_state_dict(checkpoint['state_dict'])
            if args.use_gpu:
                model.cuda()

            if "rms_global_prior" not in hypers:
                hypers["rms_global_prior"] = False

            if "dynamic_type" not in hypers:
                hypers["dynamic_type"] = "sgd"

            # If we've already estimated RMS grads for this setting, reuse them
            if hypers['dynamic'] and hypers['dynamic_type'] == 'rms':
                if hypers["rms_est_batch_size"] in rms_grad_stats and(hypers["num_timesteps"] in rms_grad_stats[hypers["rms_est_batch_size"]]):
                    for p, rms in zip(model.parameters(),
                                      rms_grad_stats[hypers["rms_est_batch_size"]][hypers["num_timesteps"]]):
                        p.RMS = rms
                else:
                    # We want to estimate the RMS of the gradients on the same number of chars every time -- how many
                    # batches do we need for the current batch size and num_timesteps
                    max_batches = int(hypers["rms_est_train_size"] /
                                      (hypers["rms_est_batch_size"] * hypers["num_timesteps"]))
                    compute_gradient_rms(model, train_data, hypers["num_timesteps"], hypers["rms_est_batch_size"],
                                         max_batches=max_batches, use_gpu=args.use_gpu)
                    if not hypers["rms_est_batch_size"] in rms_grad_stats:
                        rms_grad_stats[hypers["rms_est_batch_size"]] = dict()
                    rms_grad_stats[hypers["rms_est_batch_size"]][hypers["num_timesteps"]] =\
                        [p.RMS for p in model.parameters()]

            start = time.time()
            chars_processed, loss, losses = evaluate_rnn(model, data,
                                                        loss_function=nn.modules.loss.CrossEntropyLoss(),
                                                        num_timesteps=hypers['num_timesteps'],
                                                        use_gpu=args.use_gpu,
                                                        dynamic=hypers['dynamic'],
                                                        stats_interval=hypers['stats_interval'],
                                                        record_stats=True,
                                                        learning_rate=hypers['learning_rate'],
                                                        decay_coef=hypers['decay_coef'],
                                                        dynamic_rule=hypers["dynamic_type"],
                                                        rms_global_prior=hypers["rms_global_prior"] or None,
                                                        num_chars_to_read=hypers["eval_char_count"])

            if best_final_loss is None or loss < best_final_loss:
                best_final_loss = loss
                best_id = i

            end = time.time()
            hypers['time_min'] = (end - start) / 60
            total_time += end - start

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

            if save_images:
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

            if args.save_model:
                if hypers['dynamic']:
                    torch.save(model, os.path.join(out_dir, 'model.pth'))
                    if hypers['dynamic_type'] == 'rms':
                        torch.save([p.RMS for p in model.parameters()], os.path.join(out_dir, 'rms.pth'))

        if save_images:
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
            results['total_time_min'] = total_time / 60
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
    parser.add_argument('--debug', dest='debug', action='store_const', default=False, const=True,
                        help='If set, debugging hooks are attached to the module and check for NaN gradients')
    parser.add_argument('--save-model', dest='save_model', action='store_const', default=False, const=True,
                        help='If set, the model at the end of dynamic evaluation is stored')
    parser.add_argument('--gpu_id', default='0', type=str, help='id for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--lhuc-sparse', dest='lhuc_sparse', action='store_const', default=False, const=True,
                        help='If set, evaluation is performed with LHUC of sparse dynamic evaluation')

    args = parser.parse_args()
    if args.use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    if args.lhuc_sparse:
        evaluate_lhuc_or_sparse(args)
    else:
        evaluate(args)
