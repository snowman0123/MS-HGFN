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
        if task == 'future' or task=='trend':
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


# if __name__ == '__main__':
#     stocks=100
#     node,label=32,64
#     model = linear_attention(node,label,2)
#     x = torch.ones((66, stocks, node))
#     y = torch.ones((66, stocks, label))
#
#     out = model(x,y)
#     print(out.shape)
