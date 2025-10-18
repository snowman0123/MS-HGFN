import random
import time

from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, mean_squared_error, mean_absolute_error
from torch import nn, optim
import os

import matplotlib.pyplot as plt

from utils import ROI, Sharp, MDD
import torch

import numpy as np

torch.set_printoptions(precision=5, sci_mode=False)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch


def calculate_sharpe_ratio(asset_values, risk_free_rate=0):
    # 计算每日回报率
    returns = np.diff(asset_values) / asset_values[:-1]

    # 计算平均回报率和标准差
    average_return = np.mean(returns)
    std_deviation = np.std(returns)

    # 计算夏普率
    sharpe_ratio = (average_return - risk_free_rate) / std_deviation

    return sharpe_ratio

def daily_evaluate2(model, dataloader, A=None):
    predicts = labels = None
    model.eval()
    for data, _ in dataloader:
        # out = model(data['indicator'])
        if A != None:
            out = model(data['indicator'].squeeze(), A)
        else:
            out = model(data['indicator']).squeeze()
        out = torch.argmax(out, dim=-1).cpu().detach()
        label = data['price'].cpu().detach()
        out=out.unsqueeze(0)

        if predicts == None:
            predicts = out
            labels = label
        else:
            predicts = torch.cat([predicts, out], dim=0)
            labels = torch.cat([labels, label], dim=0)
    print(predicts.shape)
    t, n = predicts.shape
    print(t,'--------------')
    investment_price = torch.ones(t)

    for i in range(len(labels[0])):
        print(labels[:, i].shape, predicts[:, i].shape)
        investment_price += daily_simulate_trading(labels[:, i], predicts[:, i])

    investment_price=investment_price/investment_price[0]
    investment_price=(investment_price*1000000)
    investment_price=investment_price.cpu().detach().tolist()
    roi = ROI(investment_price) * 100
    print(investment_price)
    sharp = Sharp(investment_price)* 100
    mdd=MDD(investment_price)* 100

    print('ROI:{:.4f},SP:{:.4f},MDD:{:.4f}'.format(roi, sharp, mdd))

    plt.plot(investment_price, linestyle='-', color='b', label='Data')
    plt.legend()
    plt.grid(True)
    plt.show()
    return roi,sharp,mdd,investment_price


def mean_and_variance(numbers):
    mean = sum(numbers) / len(numbers)
    variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
    return mean, variance



    a, b, c, investment_price = daily_evaluate2(bmodel, test_loader, A)
    roi.append(a)
    sharp.append(b)
    mdd.append(c)
    investment_price = np.array(investment_price)
    if i == 0:
        invest = investment_price
    else:
        invest += investment_price

    print(mean_and_variance(roi))
    print(mean_and_variance(sharp))
    print(mean_and_variance(mdd))
    invest = invest / 5
    print(invest.tolist())

    plt.plot(invest, linestyle='-', color='b', label='Data')
    plt.legend()
    plt.grid(True)
    plt.show()



def daily_simulate_trading(prices, buy_signals):
    cash = 10000  # 初始资金
    stocks = 0  # 初始股票数量
    total_assets = []  # 存储每天的总资产

    for i in range(len(prices)):
        if stocks > 0 and i > 0:
            # 如果持有股票且当前不是第一天，则卖出
            cash += stocks * prices[i]
            stocks = 0
        if buy_signals[i] == 1 and stocks == 0:
            # 如果发出买入信号且没有持有股票，则买入
            stocks = cash / prices[i]
            cash -= stocks * prices[i]

        # 计算当天的总资产
        total_assets.append(cash + stocks * prices[i])

    return torch.tensor(total_assets)