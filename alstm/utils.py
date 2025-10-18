import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import torch
import os
import argparse
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
from torch import nn, optim
import os
from DataProcess import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from torch import nn, optim



def get_dimension(train_loader):
    first_batch = next(iter(train_loader))
    return first_batch[0]['indicator'][0].shape


def reset_parameters(named_parameters):
    for i in named_parameters():
        if len(i[1].size()) == 1:
            std = 1.0 / math.sqrt(i[1].size(0))
            nn.init.uniform_(i[1], -std, std)
        else:
            nn.init.xavier_normal_(i[1])

def laplacian(W):
    N, N = W.shape
    W = W + torch.eye(N).to(W.device)
    D = W.sum(axis=0)
    D = torch.diag(D ** (-0.5))
    out = D @ W @ D
    return out

