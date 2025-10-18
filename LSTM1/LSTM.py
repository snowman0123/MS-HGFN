from torch import nn
import torch
from torch.nn import Module, Parameter
import math
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")


# torch.set_printoptions(precision=4, sci_mode=False)
class time_att(nn.Module):
    def __init__(self, hidden, time_length, stocks):
        super(time_att, self).__init__()
        self.W = Parameter(torch.FloatTensor(stocks, time_length, hidden))
        nn.init.xavier_normal_(self.W.data)

    def reset_param(self, x):
        stdv = 1. / math.sqrt(x.size(1))
        x.data.uniform_(-stdv, stdv)

    def forward(self, ht):  # B T N D
        ht = torch.transpose(ht, 1, 2)
        ht_W = ht.mul(self.W)
        ht_W = torch.sum(ht_W, dim=-1)
        att = F.softmax(ht_W, dim=-1)
        return att


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
    def __init__(self, args, in_feat, out_feat, time_length, stocks):
        super(LSTM, self).__init__()
        self.in_feat = in_feat
        self.hid_size = out_feat
        self.time_length = time_length
        self.stocks = stocks
        self.lstmcell = LSTMcell(in_feat, out_feat, stocks)
        self.attention = args.attention
        self.time_att = time_att(out_feat, time_length, stocks)

    #              B*T*N*D
    def forward(self, x, hidden=None, c=None):
        h = []
        if hidden == None:
            hidden = torch.zeros((1, self.stocks, self.hid_size), device=x.device, dtype=x.dtype)
            c = torch.zeros((1, self.stocks, self.hid_size), device=x.device, dtype=x.dtype)
        for t in range(self.time_length):
            hidden, c = self.lstmcell(x[:, t], hidden, c)
            h.append(hidden)
        h = torch.stack(h, dim=-3)
        if self.attention:
            t_att = self.time_att(h)  # [B, N, T]
            h = torch.transpose(h, 1, 2)
            t_att = torch.unsqueeze(t_att, 2)
            att_ht = torch.matmul(t_att, h)
        else:
            att_ht = hidden
        return att_ht.squeeze()


class Model(nn.Module):
    def __init__(self, args, in_feat, hid_size, time_length, stocks):
        super(Model, self).__init__()
        self.in_feat = in_feat
        self.hid_size = hid_size
        self.time_length = time_length
        self.task = args.task
        self.stocktemporal = LSTM(args, in_feat, hid_size, time_length, stocks)

        self.dropout = nn.Dropout(p=.5)
        if self.task == 'price' or self.task == 'ranking':
            self.fc = nn.Linear(hid_size, 1)

    #              B*T*N*D
    def forward(self, x):
        out = None
        x = F.relu(self.stocktemporal(x))

        if self.task == 'price':
            out = F.relu(self.fc(x)).squeeze()
        if self.task == 'ranking':
            out = F.leaky_relu(self.fc(x), negative_slope=.2).squeeze()
        return out
