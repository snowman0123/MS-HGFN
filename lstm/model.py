from torch import nn
import torch
from torch.nn import Module, Parameter
import math
import torch.nn.functional as F
import warnings

from LSTM import LSTM
from Prediction import Predict

warnings.filterwarnings("ignore")


class Model(nn.Module):
    def __init__(self, in_size, hid_size, time_length, stocks, task):
        super(Model, self).__init__()

        self.in_size = in_size
        self.hid_size = hid_size

        self.lstm1 = LSTM(in_size, hid_size, time_length, stocks)
        # self.lstm2 = LSTM(hid_size, hid_size, time_length, stocks)

        self.pred = Predict(hid_size, task)

    def forward(self, x):
        temporal_embedding = self.lstm1(x)
        # temporal_embedding = self.lstm2(temporal_embedding)

        out = self.pred(temporal_embedding[:, -1])
        return out
