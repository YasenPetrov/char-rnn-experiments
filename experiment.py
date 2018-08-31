import os
import json
import argparse
import copy
import time
import traceback

import matplotlib
# We do not need to show a figure - the line below makes sure we do not look for a display to show one
matplotlib.use('Agg')

from models.rnn import get_rnn_for_hyperparams
from models.ngram import Ngram_LM
from models.utils import RNN_Hyperparameters
from training import train_rnn
from datautils.dataset import Dataset, TextFile
from utils.experiment_utils import plot_ngram_performance, experiment_setup, resume_experiment_setup
from utils.experiment_utils import on_rnn_version_end
from trainutils.trainutils import TrainLog
from utils.torch_utils import get_optimizer, get_number_of_params, load_checkpoint
from log import get_logger
from config import DEFAULT_MEMORY_LIMIT_BYTES
from utils.logging import slack_logging

logger = get_logger(__name__)

# TODO: Documentation
# TODO: Avoid downspikes in validation loss in training
# TODO: Unify interfaces of language models

# TRAIZQ
# TODO: Experiment names should start with 00>asdas
# TODO: Make hyperparam config names meaningful


def perform_experiment(args):
    logger.info('\n{0}\nStarting experiment {1}\n{2}'.format('=' * 30, args.experiment_name, '=' * 30))
    slack_logging.create_channel(args.experiment_name)

    # Parse the experiment spec file and create the necessary file structure
    spec, train_filename, valid_filename, alphabet, hyperparam_list, experiment_out_dir = experiment_setup(args)

    # Stores stats for different models - store the cwd so I know which machine I've ran this from
    results_dict = {'cwd' : os.getcwd()}
    experiment_results_filename = os.path.join(experiment_out_dir, 'results.json')

    # Create data providers
    data_train = Dataset(train_filename, alphabet, DEFAULT_MEMORY_LIMIT_BYTES)
    data_valid = Dataset(valid_filename, alphabet, DEFAULT_MEMORY_LIMIT_BYTES)

    best_valid_loss_overall = None
    ngram_stats = {'ns': [], 'deltas': [], 'valid_losses': [], 'train_losses': [], 'adapts': []}

    experiment_start = time.time()

    for i_hyperparam, hyperparams in enumerate(hyperparam_list):
        logger.info('\n{0}\n{5}: Starting training for model {3} out of {4} with hyperparameters:\n{1}\n{2}'.format(
            "-" * 50, hyperparams.to_string(), '-' * 50, i_hyperparam + 1, len(hyperparam_list), args.experiment_name
        ))
        slack_logging.send_message(args.experiment_name, slack_logging.generate_experiment_start_message(
            i_hyperparam + 1, len(hyperparam_list), hyperparams))

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
                train_results = train_rnn(
                    model=model,
                    data_train=data_train,
                    data_valid=data_valid,
                    batch_size=hyperparams.batch_size,
                    num_timesteps=hyperparams.num_timesteps,
                    hidden_state_reset_steps=hyperparams.reset_state_every,
                    num_epochs=hyperparams.num_epochs,
                    optimizer=optimizer,
                    use_gpu=args.use_gpu,
                    stats_frequency=hyperparams.batches_between_stats,
                    train_log=train_log,
                    model_checkpoints_dir=os.path.join(out_dir, 'checkpoints'),
                    train_log_file=os.path.join(out_dir, 'train_log.json'),
                    experiment_name=args.experiment_name,
                    max_grad_l2_norm=hyperparams.max_grad_l2_norm
                )
                best_model_filepath, best_model_valid_loss, mean_sec_per_batch, sec_per_batch_sd, \
                batch_count, training_time_sec = train_results

                results_dict[i_hyperparam] = {
                    'path': best_model_filepath,
                    'valid_loss': best_model_valid_loss,
                    'num_parameters': get_number_of_params(model),
                    'mean_sec_per_batch': mean_sec_per_batch,
                    'batch_count': batch_count,
                    'training_time_min': training_time_sec / 60,
                    'sec_per_batch_sd': sec_per_batch_sd,
                    'config': hyperparams.__dict__
                }

                # Update best results, save figures and logs
                best_valid_loss_overall, results_dict = on_rnn_version_end(best_valid_loss_overall, results_dict,
                                                                           i_hyperparam, hyperparams, train_log, out_dir,
                                                                           train_results, experiment_results_filename,
                                                                           args.experiment_name)

            else: # Train an n-gram model
                if not os.path.exists(experiment_out_dir):
                    os.makedirs(experiment_out_dir)

                # Train an N-gram model
                train_file_obj = TextFile(train_filename)
                valid_file_obj = TextFile(valid_filename)

                model = Ngram_LM(hyperparams.n, alphabet)

                model.train(train_file_obj)
                logger.info('{0}: Training complete. Now evaluating on training set'.format(args.experiment_name))

                # If evaluating with adaptive counts, we want the training and validation evaluations to be indep.
                model_copy = copy.deepcopy(model)

                train_loss = model_copy.evaluate(train_file_obj, hyperparams.delta, hyperparams.adapt)
                logger.info('{1}: Evaluated on training set. Loss: {0:.5f}'.format(train_loss, args.experiment_name))
                valid_loss = model.evaluate(valid_file_obj, hyperparams.delta, hyperparams.adapt)
                logger.info('{1}: Evaluated on validation set. Loss: {0:.5f}'.format(valid_loss, args.experiment_name))

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

    with open(experiment_results_filename, 'w+') as fp:
        json.dump(results_dict, fp, indent=2)

    slack_logging.upload_file(args.experiment_name,
                              **slack_logging.generate_results_message(experiment_results_filename))

    # Plot n-gram performance across hyperparameters
    if not spec['model'] == 'rnn':
        plot_ngram_performance(ngram_stats, experiment_out_dir, spec)


def resume_experiment(args):
    logger.info('\n{0}\nResuming experiment {1}\n{2}'.format('=' * 30, args.experiment_name, '=' * 30))

    results_dict, resume_spec, experiment_out_dir, alphabet, data_train, data_valid,\
        experiment_results_filename = resume_experiment_setup(args)

    slack_logging.send_message(args.experiment_name, slack_logging.generate_experiment_resume_message(resume_spec))

    best_valid_loss_overall = results_dict[results_dict['best_key']]['valid_loss']

    experiment_start = time.time()

    # Which experiments do we want to resume? All hyperparameter combinations have integer ids in the result file
    for i_hyperparam in resume_spec['ids']:
        out_dir = os.path.join(experiment_out_dir, str(i_hyperparam))

        # Load previous training results
        train_log = TrainLog.from_json(os.path.join(out_dir, 'train_log.json'))

        # Load hyperparameters
        results_dict[str(i_hyperparam)]['config']['num_epochs'] += resume_spec['num_epochs']
        config = results_dict[str(i_hyperparam)]['config']
        hyperparams = RNN_Hyperparameters(**config)

        if 'max_grad_l2_norm' in resume_spec:
            hyperparams.max_grad_l2_norm = resume_spec['max_grad_l2_norm']

        model, optimizer, train_loss_accumulator = load_checkpoint(
            out_dir, config, alphabet.get_size(), args.use_gpu, which='last')

        logger.info('\n{0}\n{5}: Resuming training training for model {3} out of {4} with hyperparameters:\n{1}\n{2}'.format(
            "-" * 50, hyperparams.to_string(), '-' * 50, i_hyperparam + 1, len(resume_spec['ids']), args.experiment_name
        ))

        slack_logging.send_message(args.experiment_name, slack_logging.generate_experiment_start_message(
            i_hyperparam + 1, len(resume_spec['ids']), hyperparams, resuming=True))

        logger.info('{1}: Training for {0} more epochs'.format(resume_spec['num_epochs'], args.experiment_name))

        # Perform training
        train_results = train_rnn(
            model=model,
            data_train=data_train,
            data_valid=data_valid,
            batch_size=hyperparams.batch_size,
            num_timesteps=hyperparams.num_timesteps,
            hidden_state_reset_steps=hyperparams.reset_state_every,
            num_epochs=hyperparams.num_epochs,
            optimizer=optimizer,
            use_gpu=args.use_gpu,
            stats_frequency=results_dict[str(i_hyperparam)]['config']['batches_between_stats'],
            train_log=train_log,
            model_checkpoints_dir=os.path.join(out_dir, 'checkpoints'),
            train_log_file=os.path.join(out_dir, 'train_log.json'),
            start_epoch=train_log.epochs[-1],
            start_batches=results_dict[str(i_hyperparam)]['batch_count'],
            start_time_sec=results_dict[str(i_hyperparam)]['training_time_min'] * 60,
            start_train_loss_accumulator=train_loss_accumulator,
            experiment_name=args.experiment_name,
            max_grad_l2_norm = hyperparams.max_grad_l2_norm
        )
        best_model_filepath, best_model_valid_loss, mean_sec_per_batch, sec_per_batch_sd, \
        batch_count, training_time_sec = train_results

        old_results = results_dict[str(i_hyperparam)]
        results_dict[str(i_hyperparam)] = {
            'path': best_model_filepath,
            'valid_loss': best_model_valid_loss,
            'num_parameters': get_number_of_params(model),
            'mean_sec_per_batch': batch_count / training_time_sec,
            'batch_count': batch_count,
            'training_time_min': training_time_sec / 60,
            'sec_per_batch_sd': 0,
            'config': hyperparams.__dict__
        }

        # Update best results, save figures and logs
        best_valid_loss_overall, results_dict = on_rnn_version_end(best_valid_loss_overall, results_dict,
                                                                   i_hyperparam, hyperparams, train_log, out_dir,
                                                                   train_results, experiment_results_filename,
                                                                   args.experiment_name)
    # Record time and save stats
    experiment_end = time.time()
    experiment_duration_min = (experiment_end - experiment_start) / 60
    results_dict['experiment_duration_min'] += experiment_duration_min

    with open(experiment_results_filename, 'w+') as fp:
        json.dump(results_dict, fp, indent=2)

    logger.info('{0}: Training complete'.format(args.experiment_name))

    slack_logging.upload_file(args.experiment_name,
                              **slack_logging.generate_results_message(experiment_results_filename))


if __name__ == '__main__':
    # TODO: Help messages
    parser = argparse.ArgumentParser(description='''Runs a training or evaluation session''')

    parser.add_argument('experiment_name', metavar='experiment_name', type=str, help='The name of the experiment')
    parser.add_argument('--gpu', dest='use_gpu', action='store_const', default=False, const=True,
                        help='If set, training is performed on a GPU, if available')
    parser.add_argument('--resume', dest='resume', action='store_const', default=False, const=True,
                        help='If set, training is continued for the specified sets of hyperparams and numbers of' +
                             'epochs specified in resume_spec.json in the experiment folder')
    parser.add_argument('--slack', dest='log_to_slack', action='store_const', default=False, const=True,
                        help='If set, some training logs are forwarded to Slack. Slack token must be set in SLACK_API_TOKEN')
    parser.add_argument('--verbose', '-v', action='count', default=1)
    parser.add_argument('--gpu_id', default='0', type=str, help='id for CUDA_VISIBLE_DEVICES')

    args = parser.parse_args()

    if args.use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    if args.log_to_slack:
        slack_logging.init_slack_client(verbosity=args.verbose)
    try:
        if args.resume:
            resume_experiment(args)
        else:
            perform_experiment(args)
    except Exception as e:
        channel_name = 'general'
        if args.experiment_name in slack_logging.g_channel_map:
            channel_name = args.experiment_name
        slack_logging.send_message(channel_name, slack_logging.generate_unexpected_error_message(traceback.format_exc(),
                                                                                              args))
        traceback.print_exc()
