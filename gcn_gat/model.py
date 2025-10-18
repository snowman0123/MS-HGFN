from torch import nn
import torch
from torch.nn import Module, Parameter
import math
import torch.nn.functional as F
import warnings

from GCN import GCN, GCNLayer
from LSTM import LSTM
from Prediction import Predict

warnings.filterwarnings("ignore")


class Model(nn.Module):
    def __init__(self, in_size, hid_size, time_length, stocks, task, gnn):
        super(Model, self).__init__()

        self.in_size = in_size
        self.hid_size = hid_size

        self.lstm = LSTM(in_size, hid_size, time_length, stocks)

        if gnn == "GCN":
            self.gnn_model = GCN(hid_size, hid_size)
        elif gnn == "GAT":
            self.gnn_model = GCNLayer(hid_size, hid_size)

        self.pred = Predict(hid_size + hid_size, task)

    def forward(self, x, adjmatrix):
        temporal_embedding = self.lstm(x)
        relational_embedding = self.gnn_model(temporal_embedding, adjmatrix)
        out = self.pred(torch.cat([temporal_embedding, relational_embedding], dim=-1))
        return out
