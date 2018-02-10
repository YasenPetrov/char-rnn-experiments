import os
import shutil
import json
import argparse
import copy
import time

import pandas as pd
import matplotlib
# We do not need to show a figure - the line below makes sure we do not look for a display to show one
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch import nn

from models.rnn import RNN_LM, get_rnn_for_hyperparams
from models.ngram import Ngram_LM
from models.utils import RNN_Hyperparameters, Ngram_Hyperparameters
from training import train_rnn, evaluate_rnn
from datautils.dataset import Alphabet, Dataset, TextFile
from utils import dict_of_lists_to_list_of_dicts
from trainutils.trainutils import TrainLog
from torchutils import get_optimizer, get_number_of_params
from log import get_logger
from config import EXPERIMENT_DIR, EXPERIMENT_SPEC_FILENAME, DATA_DIR, ALPHABETS_DIR, DEFAULT_MEMORY_LIMIT_BYTES

logger = get_logger(__name__)

# TODO: Documentation
# TODO: Avoid downspikes in validation loss in training
# TODO: Unify interfaces of language models

# TRAIZQ
# TODO: Experiment names should start with 00>asdas
# TODO: Make hyperparam config names meaningful


def train(args):
    logger.info('\n{0}\nStarting experiment {1}\n{2}'.format('=' * 30, args.experiment_name, '=' * 30))
    # We expect to find a spec.json file in a folder with the same name as the experiment under the main experiment dir
    experiment_folder = os.path.join(EXPERIMENT_DIR, args.experiment_name)

    if not os.path.exists(experiment_folder):
        raise FileNotFoundError(experiment_folder)

    spec_filepath = os.path.join(experiment_folder, EXPERIMENT_SPEC_FILENAME)

    if not os.path.exists(spec_filepath):
        raise FileNotFoundError(spec_filepath)

    with open(spec_filepath, 'r') as fp:
        spec = json.load(fp)

    # Check if data files exist
    data_folder = os.path.join(DATA_DIR, spec['data_folder'])

    train_filename = os.path.join(data_folder, 'train.txt')
    valid_filename = os.path.join(data_folder, 'valid.txt')
    for fname in [train_filename, valid_filename]:
        if not os.path.exists(fname):
            raise FileNotFoundError(fname)

    # Create alphabet
    if spec['alphabet_file'] == '':
        # No alphabet specified, create from training data
        alphabet = Alphabet.from_text(train_filename)
    else:
        # A .json file name is specified - load alphabet from there
        alphabet_file = os.path.join(ALPHABETS_DIR, spec['alphabet_file'])
        if not os.path.exists(alphabet_file):
            raise FileNotFoundError(alphabet_file)
        alphabet = Alphabet.from_json(alphabet_file)

    # Save alphabet in a .json file in the experiment home directory
    experiment_alphabet_file = os.path.join(experiment_folder, 'alphabet.json')
    alphabet.dump_to_json(experiment_alphabet_file)


    # Parse hyperparameters
    hyperparam_list = []
    for param_dict in spec['hyperparams'].values():
        for d in dict_of_lists_to_list_of_dicts(param_dict):
            if spec['model'] == 'rnn':
                hyperparam_list.append(RNN_Hyperparameters(**d))
            else:
                hyperparam_list.append(Ngram_Hyperparameters(**d))

    # Build file structure
    experiment_out_dir = os.path.join(experiment_folder, 'out')
    if os.path.exists(experiment_out_dir):
        logger.info('Removing old results directory {0}'.format(experiment_out_dir))
        shutil.rmtree(experiment_out_dir)

    # Stores stats for different models - store the cwd so I know which machine I've ran this from
    results_dict = {'cwd' : os.getcwd()}

    # Create data providers
    data_train = Dataset(train_filename, alphabet, DEFAULT_MEMORY_LIMIT_BYTES)
    data_valid = Dataset(valid_filename, alphabet, DEFAULT_MEMORY_LIMIT_BYTES)

    best_valid_loss_overall = None
    ngram_stats = {'ns': [], 'deltas': [], 'valid_losses': [], 'train_losses': [], 'adapts': []}

    experiment_start = time.time()

    for i_hyperparam, hyperparams in enumerate(hyperparam_list):
        logger.info('\n{0}\nStarting training for model {3} out of {4} with hyperparameters:\n{1}\n{2}'.format(
            "-" * 50, hyperparams.to_string(), '-' * 50, i_hyperparam + 1, len(hyperparam_list)
        ))

        try:
            if spec['model'] == 'rnn':
                # Build file structure
                out_dir = os.path.join(experiment_out_dir, str(i_hyperparam))
                os.makedirs(out_dir)

                # Save the params for visibility when inspecting results
                hyperparams.dump_to_json(os.path.join(out_dir, 'hyperparams.json'))

                # If the hidden size was specified to be invalid, we set it to the size of the alphabet
                if hyperparams.hidden_size < 1:
                    logger.info(
                        'Specified network hidden size was {0}. This will be set to the size of the alphabet which is {1}'.format(
                            hyperparams.hidden_size, alphabet.get_size()
                        ))
                    hyperparams.hidden_size = alphabet.get_size()

                # Create model
                # model = RNN_LM(hyperparams.hidden_size, alphabet.get_size(), hyperparams.batch_size,
                #                hyperparams.num_layers, hyperparams.network_type, args.use_gpu,
                #                hyperparams.recurrent_dropout,
                #                hyperparams.linear_dropout)

                model = get_rnn_for_hyperparams(hyperparams, alphabet.get_size(), args.use_gpu)

                # Create optimizer object
                optimizer_spec = hyperparams.optimizer
                if 'kwargs' not in optimizer_spec.keys():
                    optimizer_spec['kwargs'] = {}
                optimizer_spec['kwargs']['lr'] = hyperparams.learning_rate
                optimizer_spec['kwargs']['weight_decay'] = hyperparams.l2_penalty

                optimizer = get_optimizer(optimizer_spec, model)

                # Create an object to keep track of training stats
                train_log = TrainLog()

                # Perform training
                best_model_filepath, best_model_valid_loss, mean_sec_per_batch, sec_per_batch_sd,\
                batch_count, training_time_sec = train_rnn(
                    model=model,
                    data_train=data_train,
                    data_valid=data_valid,
                    batch_size=hyperparams.batch_size,
                    num_timesteps=hyperparams.num_timesteps,
                    hidden_state_reset_steps=hyperparams.reset_state_every,
                    num_epochs=hyperparams.num_epochs,
                    optimizer=optimizer,
                    use_gpu=args.use_gpu,
                    stats_frequency=spec['batches_between_stats'],
                    train_log=train_log,
                    model_checkpoints_dir=os.path.join(out_dir, 'checkpoints'),
                    train_log_file=os.path.join(out_dir, 'train_log.json')
                )

                # Keep the key for the stats of the best model in the dictionary with stats
                if best_valid_loss_overall is None or best_model_valid_loss < best_valid_loss_overall:
                    results_dict['best_key'] = str(i_hyperparam)
                    best_valid_loss_overall = best_model_valid_loss

                results_dict[i_hyperparam] = {
                    'path': best_model_filepath,
                    'valid_loss': best_model_valid_loss,
                    'num_parameters': get_number_of_params(model),
                    'mean_sec_per_batch': mean_sec_per_batch,
                    'batch_count' : batch_count,
                    'training_time_min': training_time_sec / 60,
                    'sec_per_batch_sd' : sec_per_batch_sd,
                    'config': hyperparams.__dict__
                }

                logger.info('End of training')

                # Plot losses and save image
                plt.figure(figsize=(12, 8))
                plt.grid()
                plt.plot(train_log.nums_batches_processed, train_log.train_errs, label='Train')
                plt.plot(train_log.nums_batches_processed, train_log.train_err_running_avgs, label='Train(RA)')
                plt.plot(train_log.nums_batches_processed, train_log.valid_errs, label='Valid.')
                plt.legend(fontsize=14)
                plt.xlabel('Batches processed', fontsize=14)
                plt.ylabel('BPC', fontsize=14)
                hypers_str = str(hyperparams.__dict__)
                plt.title(hypers_str[:len(hypers_str) // 2] + '\n' + hypers_str[len(hypers_str) // 2:], fontsize=10)

                plt.savefig(os.path.join(out_dir, 'losses.png'))
            else: # Train an n-gram model
                if not os.path.exists(experiment_out_dir):
                    os.makedirs(experiment_out_dir)

                # Train an N-gram model
                train_file_obj = TextFile(train_filename)
                valid_file_obj = TextFile(valid_filename)

                model = Ngram_LM(hyperparams.n, alphabet.get_size())

                model.train(train_file_obj)
                logger.info('Training complete. Now evaluating on training set')

                # If evaluating with adaptive counts, we want the training and validation evaluations to be indep.
                model_copy = copy.deepcopy(model)

                train_loss = model_copy.evaluate(train_file_obj, hyperparams.delta, hyperparams.adapt)
                logger.info('Evaluated on training set. Loss: {0:.5f}'.format(train_loss))
                valid_loss = model.evaluate(valid_file_obj, hyperparams.delta, hyperparams.adapt)
                logger.info('Evaluated on validation set. Loss: {0:.5f}'.format(valid_loss))

                ngram_stats['ns'].append(hyperparams.n)
                ngram_stats['train_losses'].append(train_loss)
                ngram_stats['deltas'].append(hyperparams.delta)
                ngram_stats['valid_losses'].append(valid_loss)
                ngram_stats['adapts'].append(hyperparams.adapt)

                results_dict[i_hyperparam] = {
                    "train_loss": train_loss,
                    "valid_loss": valid_loss
                }

                if best_valid_loss_overall is None or valid_loss < best_valid_loss_overall:
                    best_valid_loss_overall = valid_loss
                    results_dict['best'] = {
                        'valid_loss': best_valid_loss_overall,
                        'index': i_hyperparam,
                        'hyperparams': hyperparams.__dict__
                    }

        except Exception:
            logger.exception('Something went wrong. Will try to train with the rest of the hyperparameter profiles')

    # Record time and save stats
    experiment_end = time.time()
    experiment_duration_min = (experiment_end - experiment_start) / 60
    results_dict['experiment_duration_min'] = experiment_duration_min

    best_stats_filename = os.path.join(experiment_out_dir, 'results.json')
    with open(best_stats_filename, 'w+') as fp:
        json.dump(results_dict, fp, indent=2)

    # Plot n-gram performance across hyperparameters
    if not spec['model'] == 'rnn':
        stats = pd.DataFrame(ngram_stats)
        stats.to_csv(os.path.join(experiment_out_dir, 'stats.csv'), index=False)
        for key, other in [('deltas', 'ns'), ('ns', 'deltas')]:
            if spec['plot_{0}'.format(key)]:
                out_folder_name = os.path.join(experiment_out_dir, key)
                if not os.path.exists(out_folder_name):
                    os.makedirs(out_folder_name)
                for kv in stats[key].unique():
                    subset = stats[stats[key] == kv]
                    others_adapt = subset[subset['adapts']][other]
                    others_not_adapt = subset[~subset['adapts']][other]

                    train_losses_adapt = subset[subset['adapts']]['train_losses']
                    train_losses_not_adapt = subset[~subset['adapts']]['train_losses']
                    valid_losses_adapt = subset[subset['adapts']]['valid_losses']
                    valid_losses_not_adapt = subset[~subset['adapts']]['valid_losses']

                    # Plot losses and save image
                    plt.figure(figsize=(12, 8))
                    plt.grid()
                    plt.plot(others_adapt, train_losses_adapt, label='Train(adapt)')
                    plt.plot(others_adapt, valid_losses_adapt, label='Valid.(adapt)')
                    plt.plot(others_not_adapt, train_losses_not_adapt, label='Train')
                    plt.plot(others_not_adapt, valid_losses_not_adapt, label='Valid.')
                    plt.legend(fontsize=14)
                    plt.xlabel(other, fontsize=14)
                    plt.ylabel('BPC', fontsize=14)
                    plt.savefig(os.path.join(out_folder_name, '{0}.png'.format(kv)))
                    plt.close()


if __name__ == '__main__':
    # TODO: Help messages
    parser = argparse.ArgumentParser(description='''Runs a training or evaluation session''')

    parser.add_argument('experiment_name', metavar='experiment_name', type=str, help='The name of the experiment')
    parser.add_argument('--gpu', dest='use_gpu', action='store_const', default=False, const=True,
                        help='If set, training is performed on a GPU, if available')

    args = parser.parse_args()
    train(args)