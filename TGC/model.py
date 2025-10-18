from torch import nn
import torch
from torch.nn import Module, Parameter
import math
import torch.nn.functional as F
import warnings

from LSTM import LSTM
from Prediction import Predict

warnings.filterwarnings("ignore")


class TGC(nn.Module):
    def __init__(self, in_size, hid_size, time_length, stocks, task):
        super(TGC, self).__init__()

        self.in_size = in_size
        self.hid_size = hid_size

        self.temporal = LSTM(in_size, hid_size, time_length, stocks)
        self.relational=GraphModule(hid_size,hid_size)
        self.pre = Predict(hid_size, task)

    def forward(self, x, adjmatrix):
        # print('xxx',x.shape)
        temporal_embedding = self.temporal(x)
        if len(temporal_embedding.shape) == 2:
            temporal_embedding = temporal_embedding.unsqueeze(0)  # Add batch dimension if needed
        relational_embedding = self.relational(temporal_embedding, adjmatrix)
        out = self.pre(relational_embedding)

        return out


class GraphModule(nn.Module):
    def __init__(self, infeat, outfeat):
        super().__init__()

        self.g = nn.Linear(infeat + infeat + 1, 1)
        self.weight = Parameter(torch.FloatTensor(infeat, outfeat))
        self.bias = Parameter(torch.FloatTensor(outfeat))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    #                 BND
    def forward(self, inputs, relation):
        # print(f"Inputs shape: {inputs.shape}")
        b, n, d = inputs.shape[0], inputs.shape[1], inputs.shape[2]

        # 扩展 x 以匹配维度 (n*n*d)
        x_expanded = inputs.unsqueeze(1).expand(b, n, n, d)
        x_expanded_transposed = x_expanded.transpose(1, 2)

        # 将节点之间的关系与特征拼接在一起，得到形状为 (b,n, n, 2*d+1) 的结果
        relation=relation.unsqueeze(-1).unsqueeze(0).expand(b,n,n,1)

        out = torch.cat((x_expanded, x_expanded_transposed,
                         relation), dim=3)
        res = self.g(out).squeeze()
        output = torch.matmul(torch.matmul(res, inputs), self.weight) + self.bias
        return output



if __name__ == '__main__':
    pass
    # in_feat, out_feat, time_length, stocks = 64, 32, 10, 100
    # model = Student(in_feat, out_feat, time_length, stocks, 'GCN', 'trend')
    # x = torch.ones((66, time_length, stocks, in_feat))
    # out = model(x, torch.ones(66, stocks, stocks))
    # print(out[0].shape)

    # in_feat, out_feat, time_length, stocks = 64, 32, 10, 100
    # model = LinearAttention(in_feat, in_feat, in_feat)
    # x = torch.ones((66, stocks, in_feat))
    # out = model(x, x)
    # print(out.shape)
