import torch
import torch.nn as nn
from torch.autograd import Variable


class RNN_LM(nn.Module):
    #TODO: Document this class
    def __init__(self, hidden_dim, alphabet_size, batch_size=1, num_layers=1, network_type='lstm', use_gpu=True,
                 recurrent_dropout=0, linear_dropout=0):
        super(RNN_LM, self).__init__()

        if network_type not in ['vanilla', 'lstm', 'gru']:
            raise ValueError('"Argument "network_type" must be one of ["vanilla", "lstm", "gru"]')

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.network_type = network_type
        self.use_gpu = use_gpu
        self.alphabet_size = alphabet_size
        self.recurrent_dropout = recurrent_dropout

        self.rnn = self._create_network()

        self.hidden2out = nn.Linear(hidden_dim, alphabet_size)
        self.hidden = self.init_hidden(self.batch_size)

    def _create_network(self):
        kwargs = dict(
            input_size=self.alphabet_size,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.recurrent_dropout
        )
        if self.network_type == 'vanilla':
            return nn.RNN(**kwargs)
        elif self.network_type == 'lstm':
            return nn.LSTM(**kwargs)
        elif self.network_type == 'gru':
            return nn.GRU(**kwargs)

    def init_hidden(self, batch_size=None, init_values=None):
        if self.network_type == 'lstm':
            if init_values is not None:
                states = (Variable(init_values[0], requires_grad=False),
                          Variable(init_values[1], requires_grad=False))
            else:
                if batch_size is None:
                    batch_size = self.batch_size
                states = (Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim), requires_grad=False),
                          Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim), requires_grad=False))
            if self.use_gpu:
                return tuple(s.cuda() for s in states)
            else:
                return states
        else:   # GRU of Vanilla RNN - only one hidden state
            if init_values is not None:
                state = Variable(init_values, requires_grad=False)
            else:
                if batch_size is None:
                    batch_size = self.batch_size
                state = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim), requires_grad=False)

            if self.use_gpu:
                return state.cuda()
            else:
                return state

    def get_hidden_data(self):
        if self.network_type == 'lstm':
            return tuple(s.data for s in self.hidden)
        else:
            return self.hidden.data

    def forward(self, inputs):
        rnn_out, self.hidden = self.rnn(inputs, self.hidden)
        logits = self.hidden2out(rnn_out)

        return logits


def get_rnn_for_hyperparams(hyperparams, alphabet_size, use_gpu):
    model = RNN_LM(hyperparams.hidden_size, alphabet_size, hyperparams.batch_size,
                   hyperparams.num_layers, hyperparams.network_type, use_gpu,
                   hyperparams.recurrent_dropout,
                   hyperparams.linear_dropout)
    return model