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


# def get_dimension(train_loader):
#     first_batch = next(iter(train_loader))
#     return first_batch[0][0].shape

def get_dimension(train_loader):
    first_batch = next(iter(train_loader))
    return first_batch[0][0].shape



def reset_parameters(named_parameters):
    for i in named_parameters():
        if len(i[1].size()) == 1:
            std = 1.0 / math.sqrt(i[1].size(0))
            nn.init.uniform_(i[1], -std, std)
        else:
            nn.init.xavier_normal_(i[1])



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


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


def calculate_annual_sharpe_ratio(daily_returns, trading_days=252):
    """
    计算年化夏普率
    daily_returns: list 或 numpy array
    trading_days: 默认一年252个交易日
    """
    daily_returns = np.array(daily_returns)
    mean_return = np.mean(daily_returns)
    std_return = np.std(daily_returns)

    if std_return == 0:
        return 0.0
    sharpe_ratio = (mean_return / std_return) * np.sqrt(trading_days)
    return sharpe_ratio