import torch
import torch.nn as nn
from torch.autograd import Variable


class BasicRnn(nn.Module):
    #TODO: Document this class
    def __init__(self, hidden_dim, alphabet_size, batch_size=1, num_layers=1, network_type='lstm', use_gpu=True,
                 recurrent_dropout=0, linear_dropout=0):
        super(BasicRnn, self).__init__()

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

    def init_hidden(self, batch_size=None, init_values=None, requires_grad=False):
        if self.network_type == 'lstm':
            if init_values is not None:
                states = (Variable(init_values[0], requires_grad=requires_grad),
                          Variable(init_values[1], requires_grad=requires_grad))
            else:
                if batch_size is None:
                    batch_size = self.batch_size
                states = (Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim), requires_grad=requires_grad),
                          Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim), requires_grad=requires_grad))
            if self.use_gpu:
                return tuple(s.cuda() for s in states)
            else:
                return states
        else:   # GRU of Vanilla RNN - only one hidden state
            if init_values is not None:
                state = Variable(init_values, requires_grad=requires_grad)
            else:
                if batch_size is None:
                    batch_size = self.batch_size
                state = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim), requires_grad=requires_grad)

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


class StepRnn(nn.Module):
    #TODO: Document this class
    def __init__(self, hidden_dim, alphabet_size, batch_size=1, num_layers=1, network_type='lstm', use_gpu=True,
                 recurrent_dropout=0, linear_dropout=0):
        super(StepRnn, self).__init__()
        self.rnn_cell = None
        if network_type == 'vanilla':
            self.rnn_cell = nn.RNNCell(input_size=alphabet_size, hidden_size=hidden_dim)
        elif network_type == 'lstm':
            self.rnn_cell = nn.LSTMCell(input_size=alphabet_size, hidden_size=hidden_dim)
        elif network_type == 'gru':
            self.rnn_cell = nn.GRUCell(input_size=alphabet_size, hidden_size=hidden_dim)

        self.hidden_dim =  hidden_dim
        self.alphabet_size = alphabet_size
        self.network_type = network_type
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.hidden = self.init_hidden(self.batch_size)
        self.hidden2out = nn.Linear(self.hidden_dim, self.alphabet_size)
        # self.lhuc_scalers = self.init_hidden(self.batch_size)

    def init_hidden(self, batch_size=None, init_values=None):
        if self.network_type == 'lstm':
            if init_values is not None:
                states = (Variable(init_values[0], requires_grad=False),
                          Variable(init_values[1], requires_grad=False))
            else:
                if batch_size is None:
                    batch_size = self.batch_size
                states = (Variable(torch.zeros(batch_size, self.hidden_dim), requires_grad=False),
                          Variable(torch.zeros(batch_size, self.hidden_dim), requires_grad=False))
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
                state = Variable(torch.zeros(batch_size, self.hidden_dim), requires_grad=False)

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
        '''
        :param inputs: shape is num_timesteps X batch_size X alphabet_size
        :return:
        '''
        rnn_out = []
        for timestep in range(inputs.size(1)):
            self.hidden = self.rnn_cell(inputs[:, timestep, :], self.hidden)
            if self.network_type == 'lstm':
                out = self.hidden2out(self.hidden[0])
            else:
                out = self.hidden2out(self.hidden)
            rnn_out.append(out)

        return torch.stack(rnn_out, 1)


class StepRnnLhuc(StepRnn):
    def __init__(self, *args, **kwargs):
        super(StepRnnLhuc, self).__init__(*args, **kwargs)
        self.lhuc_scalers = self.init_lhuc_scalers()
        # Make sure we can get the grad w.r.t. the scalers
        self.lhuc_scalers.retain_grad()
        self.sig = nn.Sigmoid()
        self.lhuc_nonlinearity = lambda x: 2 * self.sig(x)

    def init_lhuc_scalers(self, batch_size=None, init_values=None):
        if init_values is not None:
            state = Variable(init_values, requires_grad=True)
        else:
            if batch_size is None:
                batch_size = self.batch_size
            state = Variable(torch.zeros(batch_size, self.hidden_dim), requires_grad=True)

        if self.use_gpu:
            return state.cuda()
        else:
            return state

    def forward(self, inputs):
        '''
        :param inputs: shape is num_timesteps X batch_size X alphabet_size
        :return:
        '''
        rnn_out = []
        for timestep in range(inputs.size(1)):
            self.hidden = self.rnn_cell(inputs[:, timestep, :], self.hidden)
            # Scale with a(scalers), where a is sigmoid with amplitude 2 in this case
            if self.network_type == 'lstm':
                hidden_lhuc = self.hidden[0] *self.lhuc_nonlinearity(self.lhuc_scalers)
            else:
                hidden_lhuc = self.hidden * self.lhuc_nonlinearity(self.lhuc_scalers)
            out = self.hidden2out(hidden_lhuc)
            rnn_out.append(out)

        return torch.stack(rnn_out, 1)

class RNN_LM(BasicRnn):
    def __init__(self, *args, **kwargs):
        super(RNN_LM, self).__init__(*args, **kwargs)

def get_rnn_for_hyperparams(hyperparams, alphabet_size, use_gpu):
    model = RNN_LM(hyperparams.hidden_size, alphabet_size, hyperparams.batch_size,
                   hyperparams.num_layers, hyperparams.network_type, use_gpu,
                   hyperparams.recurrent_dropout,
                   hyperparams.linear_dropout)
    model.rnn.flatten_parameters()
    return model