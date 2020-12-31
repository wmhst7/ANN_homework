import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def init(self, batch_size, device):
        #return the initial state
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def forward(self, incoming, state): # state->h_t-1, incoming->X_t
        # flag indicates whether the position is valid. 1 for valid, 0 for invalid.
        output = (self.input_layer(incoming) + self.hidden_layer(state)).tanh()
        new_state = output # stored for next step
        return output, new_state


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # TODO START
        # intialize weights and layers
        self.linear_Wz = nn.Linear(input_size, hidden_size, bias=False)
        self.linear_Uz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_Wr = nn.Linear(input_size, hidden_size, bias=False)
        self.linear_Ur = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_Wc = nn.Linear(input_size, hidden_size, bias=False)
        self.linear_Uc = nn.Linear(hidden_size, hidden_size, bias=False)

        # TODO END

    def init(self, batch_size, device):
        # TODO START
        # return the initial state
        h = torch.zeros(size=(batch_size, self.hidden_size), device=device, dtype=torch.float)
        return h
        # TODO END

    def forward(self, incoming, state):
        # TODO START
        # calculate output and new_state
        prev_h = state
        z = torch.sigmoid(self.linear_Wz(incoming) + self.linear_Uz(prev_h))
        r = torch.sigmoid(self.linear_Wr(incoming) + self.linear_Ur(prev_h))
        h_ = torch.tanh(self.linear_Wc(incoming) + self.linear_Uz(r * prev_h))
        h = (1. - z) * prev_h + z * h_

        output = h
        new_state = h
        return output, new_state
        # TODO END


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # TODO START
        # intialize weights and layers
        self.linear_Wi = nn.Linear(input_size, hidden_size)
        self.linear_Ui = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_Wo = nn.Linear(input_size, hidden_size)
        self.linear_Uo = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_Wf = nn.Linear(input_size, hidden_size)
        self.linear_Uf = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_Wc = nn.Linear(input_size, hidden_size)
        self.linear_Uc = nn.Linear(hidden_size, hidden_size, bias=False)
        # TODO END

    def init(self, batch_size, device):
        # TODO START
        # return the initial state (which can be a tuple)
        h = torch.zeros(size=(batch_size, self.hidden_size), device=device, dtype=torch.float)
        c = torch.zeros(size=(batch_size, self.hidden_size), device=device, dtype=torch.float)
        return h, c
        # TODO END

    def forward(self, incoming, state):
        # TODO START
        # calculate output and new_state
        prev_h, prev_c = state
        i = torch.sigmoid(self.linear_Wi(incoming) + self.linear_Ui(prev_h))
        f = torch.sigmoid(self.linear_Wf(incoming) + self.linear_Uf(prev_h))
        o = torch.sigmoid(self.linear_Wo(incoming) + self.linear_Uo(prev_h))
        c_ = torch.tanh(self.linear_Wc(incoming) + self.linear_Uc(prev_h))

        new_c = f * prev_c + i * c_
        new_h = o * torch.tanh(new_c)
        output = new_h
        return output, (new_h, new_c)
        # TODO END