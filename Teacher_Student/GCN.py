from torch import nn
import torch
from torch.nn import Module, Parameter
import math
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")


class GCN(Module):
    def __init__(self, in_size, hid_size):
        super(GCN, self).__init__()
        self.in_features = in_size
        self.out_features = hid_size
        self.weight = Parameter(torch.FloatTensor(in_size, hid_size))
        self.bias = Parameter(torch.zeros(hid_size))
        nn.init.xavier_uniform_(self.weight)


    def forward(self, x, adj):
        support = torch.matmul(adj, x)
        out = torch.matmul(torch.unsqueeze(support, dim=-2), torch.unsqueeze(self.weight, dim=0)).squeeze()
        return out + self.bias


class Relation_Attention(nn.Module):
    """
    calculating the importance of each pre-defined relation.
    """

    def __init__(self, in_features, out_features, dropout):
        super(Relation_Attention, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data)

    def forward(self, inputdata):
        attention_temp = torch.matmul(inputdata, self.W)
        attention = torch.relu(torch.matmul(attention_temp, torch.transpose(attention_temp, -2, -1)))
        return attention


class Self_attention(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(Self_attention, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features

        self.Q_fc = nn.Linear(in_features, out_features)
        self.K_fc = nn.Linear(in_features, out_features)
        self.V_fc = nn.Linear(in_features, out_features)

    def forward(self, inputdata):
        Q = self.Q_fc(inputdata)
        K = self.K_fc(inputdata)
        V = self.V_fc(inputdata)
        Scores = torch.matmul(Q, torch.transpose(K, -2, -1))
        Scores_softmax = F.softmax(Scores, dim=1)
        Market_Signals = torch.matmul(Scores_softmax, V)
        return Market_Signals


class GCNLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout=.2):
        super(GCNLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.attribute = AttributeGate(in_features, out_features, dropout)
        self.self_attention = Self_attention(in_features, out_features, dropout)
        self.relation_attention_ind = Relation_Attention(out_features, out_features, dropout)
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data)

    def forward(self, inputdata,A=None):
        # bnd
        market_Signals = self.self_attention(inputdata)
        # market_Signals = F.dropout(market_Signals, self.dropout, training=self.training)
        relation = self.relation_attention_ind(market_Signals)

        H = torch.matmul(torch.matmul(relation, market_Signals), self.W)

        return H


class AttributeGate(nn.Module):
    """Gate Mechanism for attribute passing"""

    def __init__(self, in_features, out_features, dropout):
        super(AttributeGate, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.empty(size=(2 * in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data)
        self.b = nn.Parameter(torch.empty(size=(1, out_features)))
        nn.init.xavier_uniform_(self.b.data)

    def forward(self, h):
        attribute_gate = self._prepare_attentional_mechanism_input(h)
        attribute_gate = torch.tanh(attribute_gate.add(self.b))
        attribute_gate = F.dropout(attribute_gate, self.dropout, training=self.training)
        return attribute_gate

    def _prepare_attentional_mechanism_input(self, h):
        input_l = torch.matmul(h, self.W[:self.in_features, :])
        input_r = torch.matmul(h, self.W[self.in_features:, :])
        return input_l.unsqueeze(1) + input_r.unsqueeze(2)


if __name__ == '__main__':
    in_feat, out_feat, time_length, stocks = 64, 32, 10, 100
    model = AttributeGate(in_feat, out_feat, dropout=.5)
    x = torch.ones((66, stocks, in_feat))
    out = model(x)
    print(out.shape)
