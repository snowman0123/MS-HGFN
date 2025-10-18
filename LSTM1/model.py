from torch import nn
import torch
from torch.nn import Module, Parameter
import math
import torch.nn.functional as F
import warnings

from Teacher_Student.GCN import GCN, GCNLayer
from Teacher_Student.LSTM import LSTM
from Teacher_Student.Prediction import Predict

warnings.filterwarnings("ignore")


class Student(nn.Module):
    def __init__(self, in_size, hid_size, time_length, stocks, gnn, task):
        super(Student, self).__init__()

        self.in_size = in_size
        self.hid_size = hid_size

        self.lstm = LSTM(in_size, hid_size, time_length, stocks)
        if gnn == "GCN":
            self.gnn_model = GCNLayer(hid_size, hid_size, stocks)
        elif gnn == "GAT":
            pass
        else:
            raise Exception("gnn mode error!")

        self.pred = Predict(hid_size, task)

    def forward(self, x, adjmatrix):
        temporal_embedding = self.lstm(x)
        relational_embedding = self.gnn_model(temporal_embedding, adjmatrix)
        out = self.pred(relational_embedding)

        return out, relational_embedding




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
        attention_weight = F.softmax(attention_score * tau, dim=1)
        z = torch.matmul(attention_weight, V)
        return z


class Teacher(nn.Module):
    def __init__(self, in_size, hid_size, time_length, stocks, gnn, task,T):
        super(Teacher, self).__init__()
        self.in_size = in_size
        self.hid_size = hid_size
        self.futureencoding = nn.Linear(T, hid_size)
        self.lstm = LSTM(in_size, hid_size, time_length, stocks)
        self.attentive = LinearAttention(hid_size, hid_size, hid_size)
        self.lamb=nn.Parameter(torch.ones(1,1))
        if gnn == "GCN":
            self.gnn_model = GCN(hid_size, hid_size)
        elif gnn == "GAT":
            self.gnn_model = GCNLayer(hid_size, hid_size)
        else:
            raise Exception("gnn mode error!")

        self.pred = Predict(hid_size, task)

    #                                bnt
    def forward(self, x, future, adjmatrix):
        temporal_embedding = self.lstm(x)
        relational_embedding = self.gnn_model(temporal_embedding, adjmatrix)

        fufeature = self.futureencoding(future)
        relational_embedding = self.attentive(temporal_embedding+(self.lamb)*relational_embedding, fufeature)
        #relational_embedding = self.attentive(relational_embedding, fufeature)

        out = self.pred(relational_embedding)
        return out, relational_embedding


if __name__ == '__main__':
    # in_feat, out_feat, time_length, stocks = 64, 32, 10, 100
    # model = Student(in_feat, out_feat, time_length, stocks, 'GCN', 'trend')
    # x = torch.ones((66, time_length, stocks, in_feat))
    # out = model(x, torch.ones(66, stocks, stocks))
    # print(out[0].shape)

    in_feat, out_feat, time_length, stocks = 64, 32, 10, 100
    model = LinearAttention(in_feat, in_feat,in_feat)
    x = torch.ones((66, stocks, in_feat))
    out = model(x, x)
    print(out.shape)
