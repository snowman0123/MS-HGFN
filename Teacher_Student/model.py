from torch import nn
import torch
from torch.nn import Module, Parameter
import math
import torch.nn.functional as F
import warnings
import random
from GCN import GCN, GCNLayer
from LSTM import LSTM
from Prediction import Predict

warnings.filterwarnings("ignore")


class Student(nn.Module):
    def __init__(self, in_size, hid_size, time_length, stocks, gnn, task):
        super(Student, self).__init__()

        self.in_size = in_size
        self.hid_size = hid_size
        self.lamb = nn.Parameter(torch.ones(1, 1))

        self.lstm = LSTM(in_size, hid_size, time_length, stocks)
        if gnn == "GCN":
            self.gnn_model = GCN(hid_size, hid_size)
        elif gnn == "GAT":
            self.gnn_model = GCNLayer(hid_size, hid_size, stocks)
        else:
            raise Exception("gnn mode error!")

        # self.pred = Predict(hid_size, task)

    def forward(self, x, adjmatrix, tmodel):
        temporal_embedding = self.lstm(x)
        relational_embedding = self.gnn_model(temporal_embedding, adjmatrix)
        out = tmodel.pred(temporal_embedding + (self.lamb) * relational_embedding)
        return out, relational_embedding

        # temporal_embedding = self.lstm(x)
        # relational_embedding = self.gnn_model(temporal_embedding, adjmatrix)
        # out = self.pred(relational_embedding)
        # return out, relational_embedding


class LinearAttention(nn.Module):
    def __init__(self, node_dim, future, in_dim):
        super(LinearAttention, self).__init__()
        self.layerQ = nn.Linear(future, in_dim)
        self.layerK = nn.Linear(node_dim, in_dim)
        self.layerV = nn.Linear(node_dim, in_dim)
        self.initialize()

    def initialize(self):
        self.layerQ.reset_parameters()
        self.layerK.reset_parameters()
        self.layerV.reset_parameters()

    def forward(self, node_emb, future, tau=0.5):
        Q = self.layerQ(future)
        K = self.layerK(node_emb)
        V = self.layerV(node_emb)
        attention_score = torch.matmul(Q, K.transpose(-2, -1))
        m_I = F.softmax(attention_score * tau, dim=1)
        m_V = F.softmax(-(attention_score * tau), dim=1)
        z_I = torch.matmul(m_I, V)
        z_V = torch.matmul(m_V, V)
        return z_I, z_V


class Teacher(nn.Module):
    def __init__(self, in_size, hid_size, time_length, stocks, gnn, task, T):
        super(Teacher, self).__init__()
        self.in_size = in_size
        self.hid_size = hid_size
        self.futureencoding = nn.Linear(T, hid_size)
        self.lstm = LSTM(in_size, hid_size, time_length, stocks)
        self.attentive = LinearAttention(hid_size, hid_size, hid_size)
        self.lamb = nn.Parameter(torch.ones(1, 1))
        if gnn == "GCN":
            self.gnn_model = GCN(hid_size, hid_size)
        elif gnn == "GAT":
            self.gnn_model = GCNLayer(hid_size, hid_size, stocks)
        else:
            raise Exception("gnn mode error!")

        self.pred = Predict(hid_size, task)
        self.interv_pred = Predict(hid_size * 2, task)

    #                                bnt
    def forward(self, x, future, adjmatrix):
        temporal_embedding = self.lstm(x)
        relational_embedding = self.gnn_model(temporal_embedding, adjmatrix)
        spatio_temporal = temporal_embedding + (self.lamb) * relational_embedding

        fufeature = self.futureencoding(future)
        z_I, z_V = self.attentive(spatio_temporal, fufeature)
        lis=list(range(z_V.shape[1]))
        random.shuffle(lis)

        z_V=z_V[:,lis,:]
        pred_I = self.pred(z_I)
        pred_V = self.interv_pred(torch.cat([z_I, z_V], dim=-1))

        return pred_I, pred_V, z_I


if __name__ == '__main__':
    # in_feat, out_feat, time_length, stocks = 64, 32, 10, 100
    # model = Student(in_feat, out_feat, time_length, stocks, 'GCN', 'trend')
    # x = torch.ones((66, time_length, stocks, in_feat))
    # out = model(x, torch.ones(66, stocks, stocks))
    # print(out[0].shape)

    in_feat, out_feat, time_length, stocks = 64, 32, 10, 100
    model = LinearAttention(in_feat, in_feat, in_feat)
    x = torch.ones((66, stocks, in_feat))
    out = model(x, x)
    print(out.shape)
