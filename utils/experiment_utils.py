import os
import shutil
import json

import pandas as pd
import matplotlib.pyplot as plt

from config import EXPERIMENT_DIR, EXPERIMENT_SPEC_FILENAME, DATA_DIR, ALPHABETS_DIR, RESUME_EXPERIMENT_SPEC_FILENAME
from config import DEFAULT_MEMORY_LIMIT_BYTES
from utils.general_utils import dict_of_lists_to_list_of_dicts
from datautils.dataset import Alphabet, Dataset
from models.utils import RNN_Hyperparameters, Ngram_Hyperparameters
from utils.logging import slack_logging

from log import get_logger

logger = get_logger(__name__)


def plot_ngram_performance(ngram_stats: dict, experiment_out_dir: str, spec: dict)-> None:
    '''
    Make plots for n-gram training performance and save those (assumes appropriate directory structure)
    :param ngram_stats: A dictionary with stats to be plotted
    :param experiment_out_dir: The output directory for the experiment
    :param spec: Experiment specification
    :return: None
    '''
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


def experiment_setup(args):
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

    return spec, train_filename, valid_filename, alphabet, hyperparam_list, experiment_out_dir


def resume_experiment_setup(args):
    # We expect to find a spec.json file in a folder with the same name as the experiment under the main experiment dir
    experiment_folder = os.path.join(EXPERIMENT_DIR, args.experiment_name)

    if not os.path.exists(experiment_folder):
        raise FileNotFoundError(experiment_folder)

    spec_filepath = os.path.join(experiment_folder, EXPERIMENT_SPEC_FILENAME)
    resume_spec_filepath = os.path.join(experiment_folder, RESUME_EXPERIMENT_SPEC_FILENAME)
    experiment_out_dir = os.path.join(experiment_folder, 'out')
    experiment_results_filename = os.path.join(experiment_out_dir, 'results.json')

    if not os.path.exists(experiment_results_filename):
        logger.warning('No experiment results file found while trying to resume experiment')
        raise FileNotFoundError(experiment_results_filename)

    with open(experiment_results_filename, 'r') as fp:
        results_dict = json.load(fp)
    with open(spec_filepath, 'r') as fp:
        spec = json.load(fp)
    with open(resume_spec_filepath, 'r') as fp:
        resume_spec = json.load(fp)

    # Alphabet, data
    data_folder = os.path.join(DATA_DIR, spec['data_folder'])
    train_filename = os.path.join(data_folder, 'train.txt')
    valid_filename = os.path.join(data_folder, 'valid.txt')

    alphabet = Alphabet.from_json(os.path.join(experiment_folder, 'alphabet.json'))

    # Create data providers
    data_train = Dataset(train_filename, alphabet, DEFAULT_MEMORY_LIMIT_BYTES)
    data_valid = Dataset(valid_filename, alphabet, DEFAULT_MEMORY_LIMIT_BYTES)

    return results_dict, resume_spec, experiment_out_dir, alphabet, data_train, data_valid, experiment_results_filename


def on_rnn_version_end(best_valid_loss_overall, results_dict, i_hyperparam, hyperparams, train_log, out_dir,
                       train_results, experiment_results_filename, experiment_name):
    best_model_filepath, best_model_valid_loss, mean_sec_per_batch, sec_per_batch_sd, batch_count, \
        training_time_sec = train_results
    # Keep the key for the stats of the best model in the dictionary with stats
    if best_valid_loss_overall is None or best_model_valid_loss < best_valid_loss_overall:
        results_dict['best_key'] = str(i_hyperparam)
        best_valid_loss_overall = best_model_valid_loss

    logger.info('End of training')

    # Plot losses and save image
    plt.figure(figsize=(12, 8))
    plt.grid()
    plt.plot(list(map(lambda x: x / train_log.batches_per_epoch, train_log.nums_batches_processed)), train_log.train_errs,
             label='Train', linewidth=2)
    plt.plot(list(map(lambda x: x / train_log.batches_per_epoch, train_log.nums_batches_processed)), train_log.valid_errs,
             label='Valid.', linewidth=2)
    plt.xticks(list(range(1, max(train_log.epochs) + 1)))
    plt.legend(fontsize=16)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('BPC', fontsize=16)
    plt.title(f'{experiment_name} configuration {i_hyperparam}', fontsize=24)
    img_path = os.path.join(out_dir, 'losses.png')
    if os.path.exists(img_path):
        os.remove(img_path)
    plt.savefig(img_path)

    with open(experiment_results_filename, 'w+') as fp:
        json.dump(results_dict, fp, indent=2)

    slack_logging.upload_file(experiment_name[:slack_logging.MAX_CHANNEL_NAME_LENGTH],
                              **slack_logging.generate_plot_message(img_path))

    return best_valid_loss_overall, results_dict
