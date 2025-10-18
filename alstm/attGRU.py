from torch import nn
import torch
from torch.nn import Module, Parameter
import math
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")

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


class attGRU(nn.Module):
    def __init__(self, in_feat, out_feat, time_length, stocks,attention):
        super(attGRU, self).__init__()
        self.in_feat = in_feat
        self.hid_size = out_feat
        self.time_length = time_length
        self.stocks = stocks
        self.grucell = GRUcell(in_feat, out_feat, stocks)
        self.time_att = time_att(out_feat, time_length, stocks)

        self.attention=attention
    #              B*T*N*D
    def forward(self, x, hidden=None):
        h = []
        if hidden == None:
            hidden = torch.zeros((1, self.stocks, self.hid_size), device=x.device, dtype=x.dtype)
        for t in range(self.time_length):
            hidden = self.grucell(x[:, t], hidden)
            h.append(hidden)
        h = torch.stack(h, dim=-3)

        if self.attention:
            t_att = self.time_att(h)  # [B, N, T]
            h = torch.transpose(h, 1, 2)
            t_att = torch.unsqueeze(t_att, 2)
            att_ht = torch.matmul(t_att, h)
        else:
            return h

        return att_ht.squeeze()


# if __name__ == '__main__':
#     in_feat, out_feat, time_length, stocks = 64, 32, 10, 100
#     model = GRU(in_feat, out_feat, time_length, stocks)
#     x = torch.ones((66, time_length, stocks, in_feat))
#     out = model(x)
#     print(out.shape)
