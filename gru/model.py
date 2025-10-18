from torch import nn
import torch
from torch.nn import Module, Parameter
import math
import torch.nn.functional as F
import warnings

from GRU import GRU
from Prediction import Predict

warnings.filterwarnings("ignore")


class Model(nn.Module):
    def __init__(self, in_size, hid_size, time_length, stocks, task):
        super(Model, self).__init__()

        self.in_size = in_size
        self.hid_size = hid_size

        self.gru = GRU(in_size, hid_size, time_length, stocks)

        self.pred = Predict(hid_size, task)

    def forward(self, x):
        # print(f"Input shape to GRU: {x.shape}")
        temporal_embedding = self.gru(x)
        out = self.pred(temporal_embedding)
        return out


