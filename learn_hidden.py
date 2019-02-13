import os
import sys
import json
import re
from time import time
from itertools import product

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from datautils.dataset import Alphabet, SentenceDataset
from models.utils import RNN_Hyperparameters
from models.rnn import get_rnn_for_hyperparams

from training import evaluate_rnn_sentences

if __name__ == '__main__':
    use_gpu = True

    exp_dir = sys.argv[1]
    spec_name = sys.argv[2]

    # Load spec
    with open(os.path.join(exp_dir, 'train_hidden', f'{spec_name}.json')) as fp:
        spec = json.load(fp)

    model_id = str(spec['model_id'])

    train_files_list = spec['train_files_list']
    valid_files_list = spec['valid_files_list']
    if 'max_train_chars' in spec:
        max_train_chars = int(spec['max_train_chars'])
    else:
        max_train_chars = np.inf
    if 'max_valid_chars' in spec:
        max_valid_chars = int(spec['max_valid_chars'])
    else:
        max_valid_chars = np.inf

    batch_sizes = spec['batch_sizes']
    num_timesteps_vals = spec['num_timesteps_vals']
    lrs = spec['lrs']
    epochs = spec['epochs']
    patience = 3

    hidden_states_folder = os.path.join(exp_dir, 'out', model_id, 'init_hiddens')
    if not os.path.exists(hidden_states_folder):
        os.makedirs(hidden_states_folder)

    alphabet = Alphabet.from_json(os.path.join(exp_dir, 'alphabet.json'))

    results_file = os.path.join(exp_dir, 'out/results.json')
    with open(results_file, 'r') as fp:
        exp_results = json.load(fp)

    hyperparams = RNN_Hyperparameters(**exp_results[str(model_id)]['config'])

    model = get_rnn_for_hyperparams(hyperparams, alphabet.get_size(), use_gpu)

    checkpoint_path = exp_results[str(model_id)]['path']

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])

    if use_gpu:
        model.cuda()

    pad_input = True

    results = []

    configs = list(product(batch_sizes, num_timesteps_vals, lrs))
    for i, (batch_size, num_timesteps, lr) in enumerate(configs):

        run_res_dict = {
            'batch_size': batch_size,
            'num_timesteps': num_timesteps,
            'lr': lr,
            'epochs': epochs
        }
        print(f'Starting training for configuration {i+1}|{len(configs)}:\n{json.dumps(run_res_dict,indent=2)}\n')

        torch.manual_seed(2019)
        np.random.seed(2019)

        # Define hidden state variables to be trained
        if use_gpu:
            initial_hidden = (
            Variable(torch.zeros(model.num_layers, batch_size, model.hidden_dim).cuda(), requires_grad=True),
            Variable(torch.zeros(model.num_layers, batch_size, model.hidden_dim).cuda(), requires_grad=True))
        else:
            initial_hidden = (Variable(torch.zeros(model.num_layers, batch_size, model.hidden_dim), requires_grad=True),
                              Variable(torch.zeros(model.num_layers, batch_size, model.hidden_dim), requires_grad=True))

        if lr > 0:
            optimizer = torch.optim.SGD(initial_hidden, lr=lr)
        else:
            optimizer = torch.optim.Adam(initial_hidden)

        loss_function = nn.modules.loss.CrossEntropyLoss()

        train_data = SentenceDataset(train_files_list, alphabet)
        valid_data = SentenceDataset(valid_files_list, alphabet)

        best_initial_hidden_state = None

        hiddens = [[initial_hidden[0].cpu().data.numpy()], [initial_hidden[1].cpu().data.numpy()]]
        train_losses = []
        valid_losses = []
        val_loss_histories = []

        min_val_loss = np.inf
        # Record final validation loss
        val_loss, run_history = evaluate_rnn_sentences(model, valid_data, initial_hidden, batch_size, num_timesteps,
                                                       pad_input, loss_function, use_gpu, max_chars=max_valid_chars)
        valid_losses.append(val_loss)
        val_loss_histories.append(run_history)
        print(f'Initial Validation Loss: {val_loss:.3f}')
        min_val_loss = val_loss
        best_initial_hidden_state = (model.hidden[0].data.cpu().numpy(), model.hidden[1].data.cpu().numpy())

        for epoch_ix in range(epochs):
            start = time()
            print(f'Starting epoch {epoch_ix + 1}|{epochs}')

            for inputs, targets in train_data.get_batch_iterator(batch_size, num_timesteps, pad_input=pad_input,
                                                                 max_chars_per_file=max_train_chars):
                model.hidden = initial_hidden

                if use_gpu:
                    inputs = Variable(torch.Tensor(inputs)).cuda()
                    targets = Variable(torch.LongTensor(targets)).cuda()
                else:
                    inputs = Variable(torch.Tensor(inputs))
                    targets = Variable(torch.LongTensor(targets))

                # Reset the gradiens in prep for a new step
                model.zero_grad()
                optimizer.zero_grad()

                # Forward pass through a batch of sequences - the result is <batch_size x num_timesteps x alphabet_size> -
                # the outputs for each sequence in the batch, at every timestep in the sequence
                logits = model(inputs)
                loss = loss_function(logits.contiguous().view(-1, logits.data.shape[-1]), targets.contiguous().view(-1))
                train_losses.append(loss.data.cpu().numpy()[0])

                # Backward pass - compute gradients, propagate gradient information back through the network
                loss.backward()

                # Update hidden state
                optimizer.step()

                hiddens[0].append(initial_hidden[0].cpu().data.numpy())
                hiddens[1].append(initial_hidden[1].cpu().data.numpy())

            print(f'Finished epoch {epoch_ix + 1}|{epochs} in {(time() - start):2f} seconds')

            val_loss, run_history = evaluate_rnn_sentences(model, valid_data, initial_hidden, batch_size, num_timesteps,
                                                           pad_input, loss_function, use_gpu, max_chars=max_valid_chars)
            valid_losses.append(val_loss)
            val_loss_histories.append(run_history)
            print(f'Validation Loss: {val_loss:.3f}')
            print('-' * 10)
            # Early stopping:
            if (len(valid_losses) > patience) and np.all(
                    [(valid_losses[-i - 2] <= valid_losses[-i - 1] + 1e-3) for i in range(patience)]):
                print(f'STOPPING: Validation loss has not improved in {patience} epochs')
                break
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_initial_hidden_state = (initial_hidden[0].data.cpu().numpy().mean(1),
                                             initial_hidden[1].data.cpu().numpy().mean(1))

        run_res_dict['valid_losses'] = valid_losses
        run_res_dict['train_losses'] = train_losses
        run_res_dict['best_state'] = best_initial_hidden_state
        run_res_dict['best_val_loss'] = min_val_loss

        results.append(run_res_dict)
        print('=' * 20)

    # Save results
    for res in results:
        conf_id = str(len(os.listdir(hidden_states_folder)))
        conf_dir = os.path.join(hidden_states_folder, conf_id)
        os.makedirs(conf_dir)

        np.save(os.path.join(conf_dir, 'h.npy'), res['best_state'][0])
        np.save(os.path.join(conf_dir, 'c.npy'), res['best_state'][1])
        del (res['best_state'])

        res['train_files_list'] = train_files_list
        res['valid_files_list'] = valid_files_list
        res['train_losses'] = [float(x) for x in res['train_losses']]
        res['valid_losses'] = [float(x) for x in res['valid_losses']]

        with open(os.path.join(conf_dir, 'stats.json'), 'w+') as fp:
            json.dump(res, fp, indent=2)