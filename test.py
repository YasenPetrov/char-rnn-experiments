import codecs
import sys
import os
import numpy as np
from datautils.dataset import Alphabet, Dataset
from models.rnn import RNN_LM
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch

from datautils.dataset import Alphabet, Dataset
from trainutils.trainutils import TrainLog
from training import train_rnn

if __name__ == '__main__':
    # filename = '../data/enwik8/enwik8.txt'
    # filename_train = '../data/lotr/lotr_fotr_train_u.txt'
    # filename_train = '../data/alice29_alpha_train.9.txt'
    filename_train = '../data/alice29.txt'

    filename_val = '../data/alice29_alpha_test.1.txt'

    alph = Alphabet.from_text(filename_train)

    data_train = Dataset(filename_train, alph)
    data_val = Dataset(filename_val, alph)

    with open(filename_train) as fp:
        text = fp.read()

    alphabet_size = alph.get_size()
    hidden_dim = alphabet_size
    num_timesteps = 64
    use_gpu = True
    batch_size = 2
    hidden_state_reset_interval = 256
    plot_every = 50

    torch.manual_seed(1231)

    model = RNN_LM(alphabet_size=alphabet_size, hidden_dim=hidden_dim, batch_size=batch_size, num_layers=1,
                   use_gpu=use_gpu, network_type='lstm', recurrent_dropout=0)

    loss_function = nn.modules.loss.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train_log = TrainLog()
    losses1 = train_rnn(model, data_train, data_val, batch_size, num_timesteps, hidden_state_reset_interval, 3,
                        optimizer, True, plot_every, train_log, experiment_name='Test_experiment')

    train_log.dump_to_json('log.json')

    #
    # model = RNN_LM(alphabet_size=alphabet_size, hidden_dim=hidden_dim, batch_size=batch_size, num_layers=1,
    #                use_gpu=use_gpu, network_type='gru')
    #
    # loss_function = nn.modules.loss.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    #
    # if use_gpu:
    #     model.cuda()
    #
    # step = 0
    # losses = []
    # total_train_loss = 0
    # loss_steps = 0
    # plot_every = 50
    # hidden_state_reset_interval = 100
    #
    # for epoch_number in range(5):
    #     batch_iterator = data_train.get_batch_iterator(batch_size, num_timesteps)
    #     for inputs, targets in batch_iterator:
    #         if use_gpu:
    #             inputs = Variable(torch.Tensor(inputs)).cuda()
    #             targets = Variable(torch.LongTensor(targets)).cuda()
    #         else:
    #             inputs = Variable(torch.Tensor(inputs))
    #             targets = Variable(torch.LongTensor(targets))
    #
    #         model.zero_grad()
    #
    #         #         TODO: reset hidden state at given intervals (problems with backprop)
    #         if step % hidden_state_reset_interval == 0:
    #             model.hidden = model.init_hidden()
    #         else:
    #             old_hidden_values = model.get_hidden_data()
    #             model.hidden = model.init_hidden(batch_size=None, init_values=old_hidden_values)
    #
    #         logits = model(inputs)
    #
    #         loss = loss_function(logits.view(-1, alphabet_size), targets.view(-1))
    #         total_train_loss += loss.data[0]
    #
    #         loss.backward()
    #
    #         optimizer.step()
    #         step += 1
    #
    #         if step % plot_every == 0:
                # calculate validation loss

                # Initialize hidden states with a batch size of 1 for evaluation
                # old_hidden = model.hidden
                # model.hidden = model.init_hidden(batch_size=1)
                #
                # # TODO: Make sure we iterate through whole file when validating
                # val_iterator = data_val.get_batch_iterator(1, batch_size * num_timesteps)
                # val_loss = 0
                # v_steps = 0
                # for v_inp, v_tar in val_iterator:
                #     if use_gpu:
                #         v_inp = Variable(torch.Tensor(v_inp)).cuda()
                #         v_tar = Variable(torch.LongTensor(v_tar)).cuda()
                #     else:
                #         v_inp = Variable(torch.Tensor(v_inp))
                #         v_tar = Variable(torch.LongTensor(v_tar))
                #     v_logits = model(v_inp)
                #
                #     val_loss += loss_function(v_logits.view(-1, alphabet_size), v_tar.view(-1)).data[0]
                #     v_steps += 1
                #
                # # Set hidden states to original ones
                # model.hidden = old_hidden
                #
                # losses.append([step, total_train_loss / plot_every, val_loss / v_steps])
                # total_train_loss = 0
