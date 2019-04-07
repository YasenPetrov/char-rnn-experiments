import os
import copy
import json
import time

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

from config import RANDOM_SEED
from log import get_logger
from trainutils.trainutils import TrainLog
from utils.general_utils import update_stats_aggr
from utils.logging import slack_logging

from models.rnn import RNN_LM
from datautils.dataset import Dataset
from torch.optim import Optimizer, SGD

LOG_2E = np.log2(np.e)

logger = get_logger(__name__)


def train_rnn(model: RNN_LM, data_train: Dataset, data_valid: Dataset, batch_size: int, num_timesteps: int,
              hidden_state_reset_steps: int, num_epochs: int, optimizer: Optimizer, use_gpu: bool, stats_frequency: int,
              train_log: TrainLog, model_checkpoints_dir:str, train_log_file:str, start_epoch: int=0,
              start_batches: int=0, start_time_sec: int=0, start_train_loss_accumulator: int=0, experiment_name: str='',
              max_grad_l2_norm=float('inf')):
    # TODO: Get start stats from train_log
    # TODO: Document this
    torch.manual_seed(RANDOM_SEED)

    # Track the best model
    best_model_state_dict = None
    best_model_validation_loss = None
    best_model_filename = os.path.join(model_checkpoints_dir, 'best.pth.tar')

    # Also, keep track of checkpoints nad performance - this is a map from filenames to bpc
    checkpoints = {
        "best": {
            "filename": best_model_filename,
            "valid_loss": ""
        },
        "last": {}
    }
    checkpoints_log_filename = os.path.join(model_checkpoints_dir, 'checkpoints.json')

    # Build file tree if necessary
    if not os.path.exists(model_checkpoints_dir):
        os.makedirs(model_checkpoints_dir)

    if use_gpu:
        model.cuda()

    # Our model returns the unnormalized log probabilities, the CrossEntropyLoss uses exactly these
    loss_function = nn.modules.loss.CrossEntropyLoss()

    # Keep track of how many batches we've processed as well as training and validation losses, how many times we have
    # recorded losses(used for running average), a running average of the training loss
    total_train_loss = start_train_loss_accumulator

    # Record rolling mean and variance times for batches
    total_batches_processed, mean_batch_time, batch_time_m2 = start_batches, 0, 0

    # Make sure that the timesteps between resettings of the hidden state are a multiple of the number of timesteps
    if not hidden_state_reset_steps % num_timesteps == 0:
        logger.warn(f'Num timesteps to reset {hidden_state_reset_steps} is not a multiple of the '
                    f'times number of timesteps {num_timesteps}. This will be changed')
        hidden_state_reset_steps -= hidden_state_reset_steps % num_timesteps

    train_start_time = time.time() - start_time_sec

    # Record initial validation loss
    _, validation_loss = evaluate_rnn(model, data_valid, loss_function, num_timesteps=batch_size * num_timesteps,
                                      use_gpu=use_gpu)

    for epoch_number in range(start_epoch, num_epochs):
        # We want to start traversing the text from the beginning - get a fresh batch generator
        batch_iterator = data_train.get_batch_iterator(batch_size, num_timesteps)
        for inputs, targets in batch_iterator:
            batch_start_time = time.time()

            if use_gpu:
                inputs = Variable(torch.Tensor(inputs)).cuda()
                targets = Variable(torch.LongTensor(targets)).cuda()
            else:
                inputs = Variable(torch.Tensor(inputs))
                targets = Variable(torch.LongTensor(targets))

            # Reset the gradiens in prep for a new step
            model.zero_grad() 


            # We reset the hidden state every so often. When we do not want to do that we still need to repackage the
            # state into a new Variable so as to disassociate it from gradient information from the previous batch
            if (total_batches_processed * num_timesteps) % hidden_state_reset_steps == 0:
                model.hidden = model.init_hidden()
            else:
                old_hidden_values = model.get_hidden_data()
                model.hidden = model.init_hidden(batch_size=None, init_values=old_hidden_values)

            # Forward pass through a batch of sequences - the result is <batch_size x num_timesteps x alphabet_size> -
            # the outputs for each sequence in the batch, at every timestep in the sequence
            logits = model(inputs)

            # Compute the loss on this batch
            # We flatten the results and targets before calculating the loss(pyTorch requires 1D targets) - autograd takes
            # care of backpropagating through the view() operation
            loss = loss_function(logits.contiguous().view(-1, logits.data.shape[-1]), targets.contiguous().view(-1))
            total_train_loss += LOG_2E * loss.data[0]

            # Backward pass - compute gradients, propagate gradient information back through the network
            loss.backward()

            # Clip gradients to prevent them from exploding
            total_grad_norm = nn.utils.clip_grad_norm(model.parameters(), max_grad_l2_norm)

            # Update the trainable parameters of the model
            optimizer.step()

            batch_end_time = time.time()
            # Update running averages with the amount of time for this batch
            total_batches_processed, mean_batch_time, batch_time_m2 = update_stats_aggr(
                (total_batches_processed, mean_batch_time, batch_time_m2), batch_end_time - batch_start_time
            )

            # Log training loss after every batch
            now = time.time()
            time_elapsed = now - train_start_time
            record = TrainLog.LogRecord(epoch_number + 1, total_batches_processed, LOG_2E * loss.data[0], validation_loss,
                                        time_elapsed_sec=int(time_elapsed), total_grad_norm=total_grad_norm)
            train_log.log_record(record, logger, experiment_name=experiment_name, log=False)

            # Compute and record training and validation losses
            if total_batches_processed % stats_frequency == 0:
                #TODO: Do not choose the number of timesteps here, think of something smarter
                _, validation_loss = evaluate_rnn(model, data_valid, loss_function, num_timesteps=batch_size * num_timesteps,
                                               use_gpu=use_gpu)

                # Record average loss since last recorded, reset accumulator
                training_loss = total_train_loss / stats_frequency
                total_train_loss = 0

                # Pass training stats to the TrainLog object - it will take care of storing them and logging them to
                # appropriate channels
                now = time.time()
                time_elapsed = now - train_start_time
                record = TrainLog.LogRecord(epoch_number + 1, total_batches_processed, training_loss, validation_loss,
                                            time_elapsed_sec=int(time_elapsed), total_grad_norm=total_grad_norm)
                train_log.log_record(record, logger, experiment_name=experiment_name)

                # If we have a new best model, update bset loss store it
                if best_model_validation_loss is None or best_model_validation_loss > validation_loss:
                    best_model_validation_loss = validation_loss
                    torch.save({
                        'epoch': epoch_number + 1,
                        'valid_loss': validation_loss,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'training_loss_accumulator': total_train_loss
                    }, best_model_filename)

        # Count batches per epoch -- we read the training file online, we need to iterate over a file once to get this
        if train_log.batches_per_epoch is None:
            train_log.batches_per_epoch = total_batches_processed

        # At the end of each epoch, save a checkpoint and stats
        _, checkpoint_validation_loss = evaluate_rnn(model, data_valid, loss_function,
                                             num_timesteps=batch_size * num_timesteps,
                                             use_gpu=use_gpu)

        # Send Slack message with stats
        message = slack_logging.generate_epoch_end_message(epoch_number + 1, num_epochs,
                                                           checkpoint_validation_loss, time.time() - train_start_time)
        slack_logging.send_message(experiment_name[:slack_logging.MAX_CHANNEL_NAME_LENGTH], message)

        # Remove last model checkpoint and make a new one
        if os.path.exists(os.path.join(model_checkpoints_dir, 'epoch.{0}.pth.tar'.format(epoch_number))):
            os.remove(os.path.join(model_checkpoints_dir, 'epoch.{0}.pth.tar'.format(epoch_number)))
        checkpoint_filename = os.path.join(model_checkpoints_dir, 'epoch.{0}.pth.tar'.format(epoch_number + 1))
        torch.save({
            'epoch': epoch_number + 1,
            'valid_loss': checkpoint_validation_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'training_loss_accumulator': total_train_loss
        }, checkpoint_filename)

        if best_model_validation_loss is None or best_model_validation_loss > checkpoint_validation_loss:
            best_model_validation_loss = checkpoint_validation_loss
            torch.save({
                'epoch': epoch_number + 1,
                'valid_loss': checkpoint_validation_loss,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'training_loss_accumulator': total_train_loss
            }, best_model_filename)

        # Dump training log - if something fails during the next epoch, at least we'll have kept what we have so far
        train_log.dump_to_json(train_log_file)
        cp_log = {'filename' : checkpoint_filename, 'valid_loss': checkpoint_validation_loss}
        checkpoints['last'] = cp_log
        checkpoints['best']['valid_loss'] = best_model_validation_loss

        # Also dump the information about filenames and bpc to a file
        with open(checkpoints_log_filename, 'w+') as fp:
            json.dump(checkpoints, fp, indent=2)

    train_end_time = time.time()
    train_time_sec = train_end_time - train_start_time

    # Compute an unbiased estimate of the standard deviation of times for a batch
    batch_time_sd = batch_time_m2 / (total_batches_processed - 1)

    return best_model_filename, best_model_validation_loss, mean_batch_time, batch_time_sd, total_batches_processed,\
           train_time_sec


def evaluate_rnn(model, data, loss_function, num_timesteps, use_gpu, dynamic=False, learning_rate=0,
                 record_stats=False, stats_interval=None, decay_coef=0, remove_unknown_tokens=False,
                 initial_hidden=None, logging_freq=int(1e4),
                 dynamic_rule='sgd', rms_epsilon=2e-5, rms_global_prior=True, num_chars_to_read=np.inf,
                 record_logits=False, entropy_penalty=0):
    # In case this is done during training, we do not want to interfere with the model's hidden state - we save that now
    # and recover it at the end of evaluation
    old_hidden = model.hidden

    # We want a batch size of 1 so we can pass through the text sequentially
    if initial_hidden is None:
        model.hidden = model.init_hidden(batch_size=1)
    else:
        model.hidden = (Variable(initial_hidden[0].data.mean(1).view(model.num_layers, 1, model.hidden_dim)),
                        Variable(initial_hidden[1].data.mean(1).view(model.num_layers, 1, model.hidden_dim)))

    # Compute RMS norm
    if dynamic and dynamic_rule == 'rms' and rms_global_prior:
        for param in model.parameters():
            if not param.requires_grad:
                continue
            param.RMSNorm = param.RMS / torch.mean(param.RMS)
            if not decay_coef == 0:
                # clip the RMSnorm to 1/decay_coef in order to not have the decay_coef exceed 1
                param.RMSNorm = torch.clamp(param.RMSNorm, min=-np.inf, max=1.0 / decay_coef)

    # Get a fresh iterator, so we can make a pass through the whole text
    # TODO: Make sure we iterate through whole file when validating
    # TODO: Fix imprecision in loss calculation
    val_iterator = data.get_batch_iterator(batch_size=1, num_timesteps=num_timesteps,
                                           remove_unknown_tokens=remove_unknown_tokens,
                                           num_chars_to_read=num_chars_to_read)

    # Keep track of the total loss, the number of batches we've processed and the time elapsed
    # The last batch might have a different number of timesteps - we need to take that into account when averaging, so
    # instead of counting batches, we count characters processed
    loss_at_last_stats = 0
    tot_loss = 0
    chars_processed = 0

    if record_stats:
        stats = {'chars_processed': [], 'loss': []}
        chars_since_last_stats = 0

    if dynamic:
        original_weigths = [copy.deepcopy(p.data) for p in model.parameters()]

    if record_logits:
        logits_history = []
        targets_history = []

    if entropy_penalty is not None and not entropy_penalty == 0:
        sm = nn.Softmax(dim=0)

    for inputs, targets in val_iterator:
        # Make variables volatile - we will not backpropagate here unless we're doing dynamic evaluation
        if use_gpu:
            inputs = Variable(torch.Tensor(inputs), volatile=(not dynamic)).cuda()
            targets = Variable(torch.LongTensor(targets), volatile=(not dynamic)).cuda()
        else:
            inputs = Variable(torch.Tensor(inputs), volatile=(not dynamic))
            targets = Variable(torch.LongTensor(targets), volatile=(not dynamic))

        if dynamic:
            # Reset the gradiens in prep for a new step
            model.zero_grad()

            old_hidden_values = model.get_hidden_data()
            model.hidden = model.init_hidden(batch_size=None, init_values=old_hidden_values)

        # Forward pass
        logits = model(inputs)


        # Compute the loss on this batch
        # We flatten the results and targets before calculating the loss(pyTorch requires 1D targets) - autograd takes
        # care of backpropagating through the view() operation
        loss = loss_function(logits.contiguous().view(-1, logits.data.shape[-1]), targets.contiguous().view(-1))

        # Calculate average loss (see training function if confused by reshaping) and multiply by the number of
        # timesteps to get the sum of losses for this batch
        tot_loss += loss.data[0] * inputs.shape[1]
        chars_processed += inputs.shape[1]

        if record_logits:
            logits_history.append(logits.contiguous().view(-1, logits.data.shape[-1]).data.cpu().numpy())
            targets_history.append(targets.contiguous().view(-1).data.cpu().numpy())

        if record_stats:
            chars_since_last_stats += inputs.shape[1]
            if chars_since_last_stats >= stats_interval:
                stats['chars_processed'].append(chars_processed)
                stats['loss'].append(LOG_2E * (tot_loss - loss_at_last_stats) / chars_since_last_stats)
                if (chars_processed % logging_freq) == 0:
                    logger.info('Chars processed: {0}, Loss: {1}'.format(chars_processed,
                                                                         LOG_2E * tot_loss / chars_processed))

                loss_at_last_stats = tot_loss
                chars_since_last_stats = 0

        if dynamic:

            if entropy_penalty is not None and not entropy_penalty == 0:
                preds = sm(logits.view(-1, logits.shape[-1]))
                # Add entropy penalty to loss
                loss += -torch.mean(torch.sum(preds * torch.log(preds), dim=0)) * entropy_penalty

            # Backward pass - compute gradients, propagate gradient information back through the network
            loss.backward()

            for p, o in zip(model.parameters(), original_weigths):
                if not p.requires_grad:
                    # We might have frozen a subset of the parameters
                    continue
                if dynamic_rule == 'sgd':
                    # SGD with global prior update
                    p.data += - learning_rate * p.grad.data + decay_coef * (o - p.data)
                elif dynamic_rule == 'sgd_pen':
                    # SGD with global prior update
                    p.data += - learning_rate * (p.grad.data + decay_coef * (o - p.data))
                elif dynamic_rule == 'sgd_pen_1':
                    # SGD with global prior update
                    p.data += - learning_rate * (p.grad.data - decay_coef * (o - p.data))
                elif dynamic_rule == 'rms':
                    # RMS with (potentially an RMS) global prior update
                    decay_scalers = 1
                    if rms_global_prior:
                        decay_scalers = p.RMSNorm
                    p.data += - learning_rate * p.grad.data / (p.RMS + rms_epsilon) + decay_coef * (
                    o - p.data) * decay_scalers
                else:
                    raise ValueError(f'Invalid dynamic rule for evaluation: {dynamic_rule}')

    # Restore hidden state
    model.hidden = old_hidden

    if record_logits:
        logits_history = np.concatenate(logits_history, axis=0)
        targets_history = np.concatenate(targets_history, axis=0)

    if record_stats:
        if record_logits:
            return chars_processed, LOG_2E * (tot_loss / chars_processed), stats, logits_history, targets_history
        return chars_processed, LOG_2E * (tot_loss / chars_processed), stats

    return chars_processed, LOG_2E * (tot_loss / chars_processed)


def evaluate_rnn_lhuc_sparse(model, data, loss_function, num_timesteps, use_gpu, learning_rate=None,
                 record_stats=False, stats_interval=None, remove_unknown_tokens=False, use_in_recurrent=False,
                 initial_hidden=None, logging_freq=int(1e4), adapt_rule='lhuc', num_chars_to_read=np.inf,
                             optimizer_fn=SGD, weight_decay=0):
    # In case this is done during training, we do not want to interfere with the model's hidden state - we save that now
    # and recover it at the end of evaluation
    old_hidden = model.hidden

    # We want a batch size of 1 so we can pass through the text sequentially
    if initial_hidden is None:
        model.hidden = model.init_hidden(batch_size=1)
    else:
        model.hidden = (Variable(initial_hidden[0].data.mean(1).view(model.num_layers, 1, model.hidden_dim)),
                        Variable(initial_hidden[1].data.mean(1).view(model.num_layers, 1, model.hidden_dim)))

    # Get a fresh iterator, so we can make a pass through the whole text
    val_iterator = data.get_batch_iterator(batch_size=1, num_timesteps=num_timesteps,
                                           remove_unknown_tokens=remove_unknown_tokens,
                                           num_chars_to_read=num_chars_to_read)

    # Keep track of the total loss, the number of batches we've processed and the time elapsed
    # The last batch might have a different number of timesteps - we need to take that into account when averaging, so
    # instead of counting batches, we count characters processed
    loss_at_last_stats = 0
    tot_loss = 0
    chars_processed = 0

    if record_stats:
        stats = {'chars_processed': [], 'loss': []}
        chars_since_last_stats = 0

    if adapt_rule == 'lhuc':
        model.lhuc_scalers = model.init_lhuc_scalers(batch_size=1)
        model.lhuc_scalers.retain_grad()
        optimizer = optimizer_fn([model.lhuc_scalers], lr=learning_rate, weight_decay=weight_decay)
    elif adapt_rule == 'sparse':
        model.M = model.init_M(batch_size=1)
        model.M.retain_grad()
        optimizer = optimizer_fn([model.M], lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f'Invalid adaptation rule: {adapt_rule}')

    for inputs, targets in val_iterator:
        # Make variables volatile - we will not backpropagate here unless we're doing dynamic evaluation
        if use_gpu:
            inputs = Variable(torch.Tensor(inputs)).cuda()
            targets = Variable(torch.LongTensor(targets)).cuda()
        else:
            inputs = Variable(torch.Tensor(inputs))
            targets = Variable(torch.LongTensor(targets))

        # Repackage hidden
        model.hidden = model.init_hidden(batch_size=None, init_values=model.get_hidden_data())

        # Forward pass
        logits = model(inputs, use_in_recurrent)

        # Compute the loss on this batch
        # We flatten the results and targets before calculating the loss(pyTorch requires 1D targets) - autograd takes
        # care of backpropagating through the view() operation
        loss = loss_function(logits.contiguous().view(-1, logits.data.shape[-1]), targets.contiguous().view(-1))

        # Calculate average loss (see training function if confused by reshaping) and multiply by the number of
        # timesteps to get the sum of losses for this batch
        tot_loss += loss.data[0] * inputs.shape[1]
        chars_processed += inputs.shape[1]

        if record_stats:
            chars_since_last_stats += inputs.shape[1]
            if chars_since_last_stats >= stats_interval:
                stats['chars_processed'].append(chars_processed)
                stats['loss'].append(LOG_2E * (tot_loss - loss_at_last_stats) / chars_since_last_stats)
                if (chars_processed % logging_freq) == 0:
                    logger.info('Chars processed: {0}, Loss: {1}'.format(chars_processed,
                                                                         LOG_2E * tot_loss / chars_processed))

                loss_at_last_stats = tot_loss
                chars_since_last_stats = 0

        loss.backward()

        optimizer.step()


    # Restore hidden state
    model.hidden = old_hidden

    if record_stats:
        return chars_processed, LOG_2E * (tot_loss / chars_processed), stats

    return chars_processed, LOG_2E * (tot_loss / chars_processed)


def evaluate_rnn_sentences(model, valid_data, initial_hidden, batch_size, num_timesteps, pad_input, loss_function,
                       use_gpu, max_chars=np.inf, dynamic=False, learning_rate=0.1, decay_coef=0):
    tot_val_loss = 0
    val_chars_processed = 0
    val_loss_history = []

    if dynamic:
        original_weigths = [copy.deepcopy(p.data) for p in model.parameters()]

    for inputs, targets in valid_data.get_batch_iterator(batch_size=1, num_timesteps=num_timesteps,
                                                         pad_input=pad_input, max_chars_per_file=max_chars):
        model.hidden = (Variable(initial_hidden[0].data.mean(1).view(model.num_layers, 1, model.hidden_dim)),
                        Variable(initial_hidden[1].data.mean(1).view(model.num_layers, 1, model.hidden_dim)))

        if use_gpu:
            inputs = Variable(torch.Tensor(inputs)).cuda()
            targets = Variable(torch.LongTensor(targets)).cuda()
        else:
            inputs = Variable(torch.Tensor(inputs))
            targets = Variable(torch.LongTensor(targets))

        # Forward pass through a batch of sequences - the result is <batch_size x num_timesteps x alphabet_size> -
        # the outputs for each sequence in the batch, at every timestep in the sequence
        logits = model(inputs)

        loss = loss_function(logits.contiguous().view(-1, logits.data.shape[-1]), targets.contiguous().view(-1))

        if dynamic:
            # Backward pass - compute gradients, propagate gradient information back through the network
            loss.backward()

            # SGD with global prior update
            for p, o in zip(model.parameters(), original_weigths):
                p.data += - learning_rate * p.grad.data + decay_coef * (o - p.data)

        val_loss_history.append(loss.data[0])
        tot_val_loss += loss.data[0] * inputs.shape[1]
        val_chars_processed += inputs.shape[1]

    val_loss = tot_val_loss / val_chars_processed
    return LOG_2E * val_loss, val_loss_history