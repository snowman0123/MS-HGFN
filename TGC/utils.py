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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from torch import nn


class HSIC(nn.Module):
    def __init__(self, sigma=1.0):
        super(HSIC, self).__init__()
        self.sigma = sigma

    def rbf_kernel(self, X, sigma):
        pairwise_sq_dists = torch.cdist(X, X, p=2) ** 2
        K = torch.exp(-pairwise_sq_dists / (2 * sigma ** 2))
        return K

    # 中心化核矩阵
    def center_kernel_matrix(self, K):
        n = K.shape[0]
        H = torch.eye(n) - torch.ones((n, n)) / n
        H = H.to(K.device)
        Kc = H @ K @ H
        return Kc

    # 计算HSIC
    def forward(self, X, Y):
        K = self.rbf_kernel(X, self.sigma)
        L = self.rbf_kernel(Y, self.sigma)
        Kc = self.center_kernel_matrix(K)
        Lc = self.center_kernel_matrix(L)
        hsic_value = torch.trace(Kc @ Lc) / (X.size(0) - 1) ** 2
        return hsic_value


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


def MDD(prices):
    max_price = prices[0]
    max_drawdown = 0

    for price in prices:
        if price > max_price:
            max_price = price
        drawdown = (max_price - price) / max_price
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    return max_drawdown


def Sharp(prices, risk_free_rate=0):
    # 计算每日回报率
    returns = np.diff(prices) / prices[:-1]

    # 计算平均回报率和标准差
    average_return = np.mean(returns)
    std_deviation = np.std(returns)

    # 计算夏普率
    sharpe_ratio = (average_return - risk_free_rate) / std_deviation

    return sharpe_ratio


def ROI(prices):
    return (prices[-1] - prices[0]) / prices[0]

def mean_and_variance(numbers):
    mean = sum(numbers) / len(numbers)
    variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
    return mean, variance
