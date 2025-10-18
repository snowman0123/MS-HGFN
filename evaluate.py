import random
import time

from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, mean_squared_error, mean_absolute_error
from torch import nn, optim
import os
from DataProcess import *
import matplotlib.pyplot as plt


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


BASEPRICE = 100000
INVESTPRICE = 10000


def daily_evaluate1(model, dataloader, A=None):
    predicts = labels = None
    model.eval()
    for data, _ in dataloader:

        # out = model(data['indicator'])
        if A != None:
            out = model(data, A)
        else:
            out = model(data)
        out = torch.argmax(out, dim=-1).cpu().detach()
        label = data['price'].cpu().detach()

        if predicts == None:
            predicts = out
            labels = label
        else:
            predicts = torch.cat([predicts, out], dim=0)
            labels = torch.cat([labels, label], dim=0)

    t, n = predicts.shape
    investment_price = torch.ones(t) * 40000  # 40000用不上的钱，初始100w

    for i in range(len(labels[0])):
        investment_price += daily_simulate_trading(labels[:, i], predicts[:, i])

    print(investment_price)
    return_rate = ROI(investment_price.tolist())
    sharp_rate = Sharp(investment_price.tolist())
    res = 'ROI:{:.6f},SP:{:.4f}\n'.format(
        return_rate, sharp_rate
    )
    plt.plot(investment_price, linestyle='-', color='b', label='Data')
    plt.legend()
    plt.grid(True)
    plt.show()
    print(res, end='')


def evaluate(model, tmodel, dataloader, T, threshold, A):
    predicts = labels = None
    model.eval()
    for data, _ in dataloader:

        out, _ = model(data['indicator'], A, tmodel)
        out = torch.argmax(out, dim=-1).cpu().detach()
        label = data['price'].cpu().detach()
        if predicts == None:
            predicts = out
            labels = label
        else:
            predicts = torch.cat([predicts, out], dim=0)
            labels = torch.cat([labels, label], dim=0)

    t, n = predicts.shape
    investment_price = torch.ones(t) * 40000  # 40000用不上的钱，初始100w

    for i in range(len(labels[0])):
        investment_price += simulate_trading(labels[:, i], predicts[:, i], T, threshold)

    investment_price=investment_price.cpu().detach().tolist()
    roi = ROI(investment_price) * 100
    print(investment_price)
    sharp = Sharp(investment_price)*100
    mdd=MDD(investment_price) * 100

    print('ROI:{:.2f},SP:{:.2f},MDD:{:.2f}'.format(roi, sharp, mdd))

    plt.plot(investment_price, linestyle='-', color='b', label='Data')
    plt.legend()
    plt.grid(True)
    plt.show()
    return roi,sharp,mdd,investment_price


def find_first_greater(matrix, threshold):
    result = torch.empty(matrix.size(1))
    for i in range(matrix.size(1)):
        row = matrix[:, i]
        indices = (row >= threshold).nonzero(as_tuple=True)[0]
        if len(indices) > 0:
            result[i] = row[indices[0]]
        else:
            result[i] = row[-1]
    return result


def evaluate2(model, dataloader, threshold, A=None):
    predicts = labels = None
    for data, label in dataloader:
        label = data['futurerate']
        if predicts == None:
            labels = label
        else:
            labels = torch.cat([labels, label], dim=0)

    investment_price = [BASEPRICE]

    for i in range(len(labels)):
        selected_stocks = labels[i]
        investment_price.append(torch.mean(selected_stocks).item())

    for i in range(1, len(investment_price)):
        investment_price[i] = investment_price[i - 1] + investment_price[i] * BASEPRICE
    print((investment_price[-1] - investment_price[0]) / investment_price[0])

    plt.plot(investment_price, linestyle='-', color='b', label='Data')
    plt.legend()
    plt.grid(True)
    plt.show()
    print(investment_price)


def simulate_trading(prices, buy_signals, T, threshold):
    cash = 10000  # 初始资金
    stocks = 0  # 初始股票数量
    total_assets = []  # 存储每天的总资产
    holding_days = 0  # 持有股票的天数
    last_buy_price = 0  # 上次买入的价格

    for i in range(len(prices)):
        price = prices[i]
        if holding_days > 0:
            # 如果持有股票，则计算未来五天内的收益率
            holding_days += 1
            if holding_days < T and (price - last_buy_price) / last_buy_price >= threshold:
                # 在未来五天内收益率达到5%，则卖出股票
                cash += stocks * price
                stocks = 0
                holding_days = 0
            elif holding_days == T:
                # 如果未来五天内未达到5%的收益率，则在第五天卖出股票
                cash += stocks * price
                stocks = 0
                holding_days = 0
        if buy_signals[i] == 1 and cash > 0 and stocks == 0:
            # 如果发出买入信号，有足够的现金且没有持有股票，则买入
            stocks = cash / price
            cash -= stocks * price
            last_buy_price = price
            holding_days = 1  # 重置持有股票的天数

        total_assets.append(cash + stocks * price)

    return torch.tensor(total_assets)


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

# def daily_evaluate(model, dataloader):
#     predicts = labels = None
#     model.eval()
#     for data, _ in dataloader:
#         out = torch.argmax(model(data['indicator']).cpu().detach(), dim=-1)
#         label = data['futurerate'][:, 0]
#         if predicts == None:
#             predicts = out
#             labels = label
#         else:
#             predicts = torch.cat([predicts, out], dim=0)
#             labels = torch.cat([labels, label], dim=0)
#
#     investment_price = [BASEPRICE]
#     sharp = []
#
#     for i in range(len(predicts)):
#         pred = torch.where(predicts[i] == 1)[0]
#         selected_stocks = labels[i, pred]
#
#         investment_price.append(torch.mean(selected_stocks).item())
#
#     for i in range(1, len(investment_price)):
#         investment_price[i] = investment_price[i - 1] + investment_price[i] * INVESTPRICE
#
#         sharp.append(
#             torch.mean(selected_stocks) / torch.std(selected_stocks) if torch.abs(
#                 torch.std(selected_stocks)) > 1e-3 else torch.mean(selected_stocks) / 1e-3)
#
#     plt.plot(investment_price, linestyle='-', color='b', label='Data')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#     print(investment_price)
#
#     return_rate = (investment_price[-1] - investment_price[0]) / investment_price[0]
#     sharp_rate = calculate_sharpe_ratio(investment_price)
#     res = 'ROI:{:.6f},SP:{:.4f}\n'.format(
#         return_rate, sharp_rate
#     )
#     print(res, end='')
