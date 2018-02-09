import os
import time
import copy
import json
import time

import torch
from torch import nn
from torch.autograd import Variable

from config import RANDOM_SEED
from log import get_logger
from trainutils.trainutils import TrainLog
from utils import update_stats_aggr


logger = get_logger(__name__)


def train_rnn(model, data_train, data_valid, batch_size, num_timesteps, hidden_state_reset_steps, num_epochs,
              optimizer, use_gpu, stats_frequency, train_log, model_checkpoints_dir, train_log_file):
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
    total_train_loss = 0
    training_loss_running_average = 0
    loss_records = 0
    losses = []
    # Record rolling mean and variance times for batches
    total_batches_processed, mean_batch_time, batch_time_m2 = 0, 0, 0

    train_start_time = time.time()
    for epoch_number in range(num_epochs):
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
            if total_batches_processed * num_timesteps >= hidden_state_reset_steps:
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
            total_train_loss += loss.data[0]

            # Backward pass - compute gradients, propagate gradient information back through the network
            loss.backward()

            # Update the trainable parameters of the model
            optimizer.step()


            batch_end_time = time.time()
            # Update running averages with the amount of time for this batch
            total_batches_processed, mean_batch_time, batch_time_m2 = update_stats_aggr(
                (total_batches_processed, mean_batch_time, batch_time_m2), batch_end_time - batch_start_time
            )

            # Compute and record training and validation losses
            if total_batches_processed % stats_frequency == 0:
                #TODO: Do not choose the number of timesteps here, think of something smarter
                validation_loss = evaluate_rnn(model, data_valid, loss_function, num_timesteps=batch_size * num_timesteps,
                                               use_gpu=use_gpu)

                # Record average loss since last recorded, reset accumulator
                training_loss = total_train_loss / stats_frequency
                total_train_loss = 0

                # Update running average loss
                loss_records += 1
                training_loss_running_average = training_loss_running_average * ((loss_records - 1) / loss_records) + \
                    training_loss * (1 / loss_records)

                # Record the number of characters processed, and the losses
                losses.append([total_batches_processed * num_timesteps * batch_size, training_loss,
                               training_loss_running_average, validation_loss])

                # Pass training stats to the TrainLog object - it will take care of storing them and logging them to
                # appropriate channels
                now = time.time()
                time_elapsed = now - train_start_time
                record = TrainLog.LogRecord(epoch_number + 1, total_batches_processed, training_loss,
                                            training_loss_running_average, validation_loss,
                                            time_elapsed_sec=int(time_elapsed))
                train_log.log_record(record, logger)

                # If we have a new best model, update bset loss store it
                if best_model_validation_loss is None or best_model_validation_loss > validation_loss:
                    best_model_validation_loss = validation_loss
                    torch.save({
                        'epoch': epoch_number + 1,
                        'valid_loss': validation_loss,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict()
                    }, best_model_filename)

        # At the end of each epoch, save a checkpoint and stats
        checkpoint_validation_loss = evaluate_rnn(model, data_valid, loss_function,
                                             num_timesteps=batch_size * num_timesteps,
                                             use_gpu=use_gpu)
        # Remove last model checkpoint and make a new one
        if os.path.exists(os.path.join(model_checkpoints_dir, 'epoch.{0}.pth.tar'.format(epoch_number))):
            os.remove(os.path.join(model_checkpoints_dir, 'epoch.{0}.pth.tar'.format(epoch_number)))
        checkpoint_filename = os.path.join(model_checkpoints_dir, 'epoch.{0}.pth.tar'.format(epoch_number + 1))
        torch.save({
            'epoch': epoch_number + 1,
            'valid_loss': checkpoint_validation_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, checkpoint_filename)
        torch.save(model.state_dict(), checkpoint_filename)

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


def evaluate_rnn(model, data, loss_function, num_timesteps, use_gpu):
    # In case this is done during training, we do not want to interfere with the model's hidden state - we save that now
    # and recover it at the end of evaluation
    old_hidden = model.hidden
    # We want a batch size of 1 so we can pass through the text sequentially
    model.hidden = model.init_hidden(batch_size=1)

    # Get a fresh iterator, so we can make a pass through the whole text
    # TODO: Make sure we iterate through whole file when validating
    # TODO: Fix imprecision in loss calculation
    val_iterator = data.get_batch_iterator(batch_size=1, num_timesteps=num_timesteps)

    # Keep track of the total loss, the number of batches we've processed and the time elapsed
    # The last batch might have a different number of timesteps - we need to take that into account when averaging, so
    # instead of counting batches, we count characters processed
    tot_loss = 0
    chars_processed = 0

    for inputs, targets in val_iterator:

        start = time.time()

        if use_gpu:
            inputs = Variable(torch.Tensor(inputs)).cuda()
            targets = Variable(torch.LongTensor(targets)).cuda()
        else:
            inputs = Variable(torch.Tensor(inputs))
            targets = Variable(torch.LongTensor(targets))

        # Forward pass
        logits = model(inputs)

        # Calculate average loss (see training function if confused by reshaping) and multiply by the number of
        # timesteps to get the sum of losses for this batch
        tot_loss += loss_function(logits.view(-1, logits.data.shape[-1]), targets.view(-1)).data[0] * inputs.shape[1]
        chars_processed += inputs.shape[1]

    # Restore hidden state
    model.hidden = old_hidden

    return tot_loss / chars_processed