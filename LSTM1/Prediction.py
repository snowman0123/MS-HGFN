from torch import nn
import torch
from torch.nn import Module, Parameter
import math
import torch.nn.functional as F
import warnings


class Predict(nn.Module):
    def __init__(self, nhid, task):
        super(Predict, self).__init__()
        self.layer1 = nn.Linear(nhid, nhid, bias=True)

        if task == 'price' or task == 'ranking':
            self.pred = nn.Linear(nhid, 1)
        if task == 'future':
            self.pred = nn.Linear(nhid, 2)

        self.initialize()

    def initialize(self):
        self.layer1.reset_parameters()
        self.pred.reset_parameters()

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.pred(x).squeeze()
        return x


class linear_attention(nn.Module):
    def __init__(self, node_dim,label_dim,in_dim):
        super(linear_attention,self).__init__()
        self.layerQ = nn.Linear(label_dim, in_dim)
        self.layerK = nn.Linear(node_dim, in_dim)
        self.layerV = nn.Linear(node_dim, in_dim)
        self.initialize()

    def initialize(self):
        self.layerQ.reset_parameters()
        self.layerK.reset_parameters()
        self.layerV.reset_parameters()

    def forward(self, node_emb, future_emb, tau=0.5):
        # pdb.set_trace()
        Q = self.layerQ(future_emb)
        K = self.layerK(node_emb)
        V = self.layerV(node_emb)
        attention_score = torch.matmul(Q, K.transpose(-2, -1))
        attention_weight = F.softmax(attention_score * tau, dim=1)
        z = torch.matmul(attention_weight, V)
        return z


if __name__ == '__main__':
    stocks=100
    node,label=32,64
    model = linear_attention(node,label,2)
    x = torch.ones((66, stocks, node))
    y = torch.ones((66, stocks, label))

    out = model(x,y)
    print(out.shape)
