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
        nn.init.xavier_uniform_(self.weight)
        self.bias = Parameter(torch.zeros(hid_size))

    def forward(self, x, adj):
        support = torch.matmul(adj, x)
        out = torch.matmul(torch.unsqueeze(support, dim=-2), torch.unsqueeze(self.weight, dim=0)).squeeze()
        return out + self.bias


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.2):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        # GAT 权重参数
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        # 注意力矩阵，考虑邻接关系
        attention = torch.where(adj > 0, e, -9e15 * torch.ones_like(e))
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # 使用注意力加权求和特征
        h_prime = torch.matmul(attention, Wh)
        return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # print("Wh.shape:", Wh.shape)
        N = Wh.size()[0]
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        # print("Wh.shape:", Wh.repeat(N, 1).shape)
        Wh_repeated_alternating = Wh.repeat(N, 1)


        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.2):
        super(GCNLayer, self).__init__()
        self.gat = GATLayer(in_features, out_features, dropout, alpha)
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

    def forward(self, inputdata, adj):
        # 使用自注意力机制（GAT）计算全局邻接矩阵
        # print('adj.shape:',adj.shape)
        # print('inputdata.shape:', inputdata.shape)
        h_prime = self.gat(inputdata, adj)

        # 使用学到的邻接矩阵更新节点特征
        out = torch.matmul(adj, h_prime)
        return out
