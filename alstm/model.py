from torch import nn
import torch
from torch.nn import Module, Parameter
import math
import torch.nn.functional as F
import warnings

from Prediction import Predict
from alstm.attGRU import attGRU

warnings.filterwarnings("ignore")


class Model(nn.Module):
    def __init__(self, in_size, hid_size, time_length, stocks, task):
        super(Model, self).__init__()

        self.in_size = in_size
        self.hid_size = hid_size

        self.gru1 = attGRU(in_size, hid_size, time_length, stocks,False)
        self.gru2 = attGRU(hid_size, hid_size, time_length, stocks,True)

        self.pred = Predict(hid_size, task)

    def forward(self, x):
        temporal_embedding = self.gru1(x)
        temporal_embedding = self.gru2(temporal_embedding)

        out = self.pred(temporal_embedding)
        return out


