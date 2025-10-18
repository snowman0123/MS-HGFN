from torch import nn
import torch
from torch.nn import Module, Parameter
import math
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")


class GRUgate(nn.Module):
    def __init__(self, input_size, output_size, activation, stocks, ):
        super(GRUgate, self).__init__()
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


class GRUcell(nn.Module):
    def __init__(self, in_size, out_size, stocks):
        super(GRUcell, self).__init__()
        self.in_size = in_size
        self.out_feat = out_size
        self.update = GRUgate(in_size + out_size, out_size, nn.Sigmoid(), stocks)
        self.reset = GRUgate(in_size + out_size, out_size, nn.Sigmoid(), stocks)
        self.hat = GRUgate(in_size + out_size, out_size, nn.Tanh(), stocks)

    def forward(self, xt, hidden):  # hidden:t-1

        _, N, D = hidden.shape

        zt = self.update(torch.cat([xt, hidden.expand(len(xt), N, D)], dim=-1))
        rt = self.reset(torch.cat([xt, hidden.expand(len(xt), N, D)], dim=-1))
        h_t = self.hat(torch.cat([xt, (hidden * rt).expand(len(xt), N, D)], dim=-1))

        ht = (1 - zt) * hidden + zt * h_t
        return ht


class GRU(nn.Module):
    def __init__(self, in_feat, out_feat, time_length, stocks):
        super(GRU, self).__init__()
        self.in_feat = in_feat
        self.hid_size = out_feat
        self.time_length = time_length
        self.stocks = stocks
        self.grucell = GRUcell(in_feat, out_feat, stocks)

    #              B*T*N*D
    def forward(self, x, hidden=None):
        h = []
        # print('x',x.shape)
        if hidden == None:
            hidden = torch.zeros((1, self.stocks, self.hid_size), device=x.device, dtype=x.dtype)
        for t in range(self.time_length):
            hidden = self.grucell(x[:, t], hidden)
            h.append(hidden)
        att_ht = hidden
        return att_ht.squeeze()


# if __name__ == '__main__':
#     in_feat, out_feat, time_length, stocks = 64, 32, 10, 100
#     model = GRU(in_feat, out_feat, time_length, stocks)
#     x = torch.ones((66, time_length, stocks, in_feat))
#     out = model(x)
#     print(out.shape)
