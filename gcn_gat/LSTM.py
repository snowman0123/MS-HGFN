from torch import nn
import torch
from torch.nn import Module, Parameter
import math
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")


class LSTMgate(nn.Module):
    def __init__(self, input_size, output_size, activation, stocks, ):
        super(LSTMgate, self).__init__()
        self.activation = activation
        self.W = Parameter(torch.FloatTensor(stocks, input_size, output_size))
        self.bias = Parameter(torch.zeros(stocks, output_size))
        self.reset_param(self.W)

    def reset_param(self, x):
        stdv = 1. / math.sqrt(x.size(1))
        x.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = torch.unsqueeze(x, -2)
        return self.activation(torch.matmul(x, self.W).squeeze() + self.bias)


class LSTMcell(nn.Module):
    def __init__(self, in_size, out_size, stocks):
        super(LSTMcell, self).__init__()
        self.in_size = in_size
        self.out_feat = out_size
        self.input = LSTMgate(in_size + out_size, out_size, nn.Sigmoid(), stocks)
        self.output = LSTMgate(in_size + out_size, out_size, nn.Sigmoid(), stocks)
        self.forget = LSTMgate(in_size + out_size, out_size, nn.Sigmoid(), stocks)
        self.candidate = LSTMgate(in_size + out_size, out_size, nn.Tanh(), stocks)

    def forward(self, xt, hidden, ct_1):  # hidden:t-1
        _, N, D = hidden.shape
        it = self.input(torch.cat([xt, hidden.expand(len(xt), N, D)], dim=-1))
        ot = self.output(torch.cat([xt, hidden.expand(len(xt), N, D)], dim=-1))
        ft = self.forget(torch.cat([xt, hidden.expand(len(xt), N, D)], dim=-1))
        chat = self.candidate(torch.cat([xt, hidden.expand(len(xt), N, D)], dim=-1))

        ct = ft * ct_1.expand(len(xt), N, D) + it * chat
        ht = ot * torch.tanh(ct)
        return ht, ct


class LSTM(nn.Module):
    def __init__(self, in_feat, out_feat, time_length, stocks):
        super(LSTM, self).__init__()
        self.in_feat = in_feat
        self.hid_size = out_feat
        self.stocks = stocks
        self.lstmcell = LSTMcell(in_feat, out_feat, stocks)

    #              B*T*N*D
    def forward(self, x, hidden=None, c=None):
        h = []
        if hidden == None:
            hidden = torch.zeros((1, self.stocks, self.hid_size), device=x.device, dtype=x.dtype)
            c = torch.zeros((1, self.stocks, self.hid_size), device=x.device, dtype=x.dtype)
        for t in range(len(x[0])):
            hidden, c = self.lstmcell(x[:, t], hidden, c)
            h.append(hidden)
        att_ht = hidden
        return att_ht.squeeze()


if __name__ == '__main__':
    in_feat, out_feat, time_length, stocks = 64, 32, 10, 100
    model = LSTM(in_feat, out_feat, time_length, stocks)
    x = torch.ones((66, time_length, stocks, in_feat))
    out = model(x)
    print(out.shape)
