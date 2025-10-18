import time
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, mean_squared_error, mean_absolute_error
from torch import nn, optim
import os

from DataProcess import LoadData
from evaluate import *
from utils import *
import matplotlib.pyplot as plt
from model import *
from Prediction import *

from Labels import *
torch.set_printoptions(precision=5, sci_mode=False)


torch.set_printoptions(precision=5, sci_mode=False)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from torch import nn, optim
import torch

# 检查是否有可用的CUDA设备
if torch.cuda.is_available():
    # 如果有，选择第一个可用的CUDA设备
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    # 如果没有，则默认使用CPU
    device = torch.device("cpu")
    print("Running on the CPU")

# 接下来，您可以将此'device'变量用于模型和数据的.to(device)调用中

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda:1',
                    help='GPU To Use')

parser.add_argument('--epochs', type=int, default=20,
                    help='Training max epoch')

parser.add_argument('--lr', type=float, default=0.0005,
                    help='Learning rate')

parser.add_argument('--weight_constraint', type=float, default=5e-4,
                    help='L2 Weight Constraint')

parser.add_argument('--time_length', type=int, default=1,
                    help='time length')

parser.add_argument('--hidden_feat', type=int, default=32, )

parser.add_argument('--trend', type=str, default='ranking', )

parser.add_argument('--task', type=str, default='trend', )

parser.add_argument('--alpha', type=float, default=1., )

parser.add_argument('--dropout', type=float, default=.6, )

parser.add_argument('--batch_size', type=int, default=1)

parser.add_argument('--datanormtype', type=str, default='meanstd')

parser.add_argument('--in_size', type=int, default=5)

parser.add_argument('--T', type=int, default=1)

parser.add_argument('--expect', type=float, default=0.04, help='0.04, 0.025, 0.015')


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# def test(model, dataloader, A):
#     predicts = None
#     labels = None
#     actual_prices = []  # 添加一个列表来存储实际价格
#     model.eval()
#     for data, label in dataloader:
#         # print(data)  # 打印以审查是否数据结构正确
#         # break  # 只打印一个批次
#         model.eval()
#         out = model(data['indicator'], A)
#         out = torch.argmax(out.cpu().detach(), dim=-1).reshape(-1)
#         label = label.cpu().detach().reshape(-1)
#
#         # 假设 'data' 包含实际价格的信息
#         if 'price' in data:
#             actual_prices.extend(data['price'].cpu().detach().numpy())
#
#         if predicts is None:
#             predicts = out
#             labels = label
#         else:
#             predicts = torch.cat([predicts, out])
#             labels = torch.cat([labels, label])
#
#         # 过滤labels为空的情况
#     if labels is None:
#         print("标签获取失败，检查您的数据集和解析逻辑。")
#         return None, None, None
#
#     print("比例是:{:.2f}".format((torch.sum(torch.eq(labels, 1)) / len(labels)).item() * 100), '%', end=' ')
#     print("精度是:{:.2f}".format(accuracy_score(predicts, labels) * 100), '% ',
#           'MCC是:{:.4f}'.format(matthews_corrcoef(predicts, labels)))
#
#     # 返回预测、标签和实际价格
#     return predicts.numpy(), labels.numpy(), actual_prices  # 转换为numpy以便于后续处理


def test(model, dataloader, A):
    all_probs = []
    all_labels = []
    all_prices = []

    model.eval()
    for data, label in dataloader:
        with torch.no_grad():
            out = model(data['indicator'], A)  # shape: [batch, stocks, classes]
            probs = F.softmax(out, dim=-1)[..., 1]  # 取每个股票上涨的概率
            all_probs.append(probs.cpu().numpy())
            all_labels.append(label.cpu().numpy())
            if 'price' in data:
                all_prices.append(data['price'].cpu().numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels), np.concatenate(all_prices)



def train(args, smodel, train_loader):
    for epoch in range(args.epochs):
        avg_loss = 0
        smodel.train()
        print('epoch ' + str(epoch), end=':')
        for data, label in train_loader:
            optimizer.zero_grad()
            # print(data['indicator'].shape,data['future'].shape,label.shape)
            out = smodel(data['indicator'], A)
            _, _, c = out.shape
            out = out.reshape(-1, c)
            label = label.reshape(-1)
            loss = criterion(out, label)

            avg_loss += loss
            loss.backward()
            optimizer.step()

        avg_loss = avg_loss / len(train_loader)
        print(avg_loss)


def calculate_returns(predicts, labels, actual_prices):
    returns = []
    investment_value = 1.0  # 假设起始投资为1单位（可以为任何基准值）
    for prediction, label, actual_price in zip(predicts, labels, actual_prices):
        if prediction == 1:
            # 如果实际也是上涨
            if label == 1:
                returns.append(investment_value * actual_price)
            else:
                # 如果预测错误，考虑损失
                returns.append(-investment_value * actual_price)
        elif prediction == 0:
            if label == 0:
                returns.append(-investment_value * actual_price)
            else:
                returns.append(investment_value * actual_price)
    return returns

def calculate_topk_returns(probs, labels, prices, k=5, initial_investment=10000000):
    num_days, num_stocks = probs.shape
    investment = initial_investment
    returns = []
    daily_investments = []

    for day in range(1, num_days):
        day_probs = probs[day]
        day_labels = labels[day]
        day_prices = prices[day]
        prev_prices = prices[day - 1]

        topk_indices = np.argsort(day_probs)[-k:]

        day_return = 0
        for idx in topk_indices:
            if prev_prices[idx] == 0:
                continue

            price_change = (day_prices[idx] - prev_prices[idx]) / prev_prices[idx]

            # 预测上涨：收益，预测错误：亏损
            if labels[day][idx] == 1:
                day_return += investment * price_change / k
            else:
                day_return -= investment * price_change / k

        investment += day_return
        returns.append(day_return)
        daily_investments.append(investment)

    return returns, investment, daily_investments


def calculate_annual_sharpe_ratio(returns, risk_free_rate=0):
    if len(returns) == 0 or np.std(returns) == 0:
        return np.nan  # 或者选择合适的默认值
    mean_return = np.mean(returns)
    std_dev = np.std(returns)

    trading_days = 252  # 假设一年252个交易日
    annual_sharpe_ratio = ((mean_return - risk_free_rate) / std_dev) * np.sqrt(trading_days)

    return annual_sharpe_ratio


def calculate_daily_returns(predicts, labels, actual_prices, initial_investment=10000000):
    daily_returns = []
    previous_price = None
    daily_investment = initial_investment

    # 确保有实际价格
    if not actual_prices:
        raise ValueError("实际价格列表不能为空。")

    # 初始化 previous_price 为第一天收盘价
    previous_price = actual_prices[0]

    for prediction, label, actual_price in zip(predicts, labels, actual_prices):
        if prediction == 1:  # 预测上涨
            if label == 1:  # 实际上涨
                daily_return = daily_investment * ((actual_price - previous_price) / previous_price)
            else:  # 预测错误，损失处理
                daily_return = -daily_investment * ((actual_price - previous_price) / previous_price)
        else:  # 预测下跌
            if label == 0:  # 实际下跌
                daily_return = -daily_investment * ((actual_price - previous_price) / previous_price)
            else:
                daily_return = daily_investment * ((actual_price - previous_price) / previous_price)

        # 更新 previous_price 为当前价格
        previous_price = actual_price
        # 将每日收益加入列表
        daily_returns.append(daily_return)
        # 更新投资金额，应用收益
        daily_investment += daily_return

    return daily_returns, daily_investment

def calculate_annual_sharpe_ratio_from_returns(returns, risk_free_rate=0):
    returns = np.array(returns)
    if len(returns) == 0 or np.std(returns) == 0:
        return np.nan
    daily_mean = np.mean(returns)
    daily_std = np.std(returns)
    sharpe_ratio = (daily_mean - risk_free_rate) / daily_std * np.sqrt(252)
    return sharpe_ratio




def calculate_daily_sharpe_ratio(daily_returns, risk_free_rate=0):
    daily_returns = np.array(daily_returns)
    if len(daily_returns) == 0 or np.std(daily_returns) == 0:
        return np.nan  # Handle the edge case where standard deviation is zero
    mean_daily_return = np.mean(daily_returns - risk_free_rate)
    std_dev = np.std(daily_returns)

    daily_sharpe_ratio = (mean_daily_return / std_dev)
    return daily_sharpe_ratio

def save_daily_results_to_file(daily_returns, daily_sharpe_ratios, file_path='daily_results.txt'):
    with open(file_path, 'w') as file:
        for i, (daily_return, daily_sharpe) in enumerate(zip(daily_returns, daily_sharpe_ratios)):
            file.write(f"Day {i + 1}:\n")
            file.write(f"  Daily Return: {daily_return}\n")
            file.write(f"  Daily Sharpe Ratio: {daily_sharpe}\n")
            file.write("\n")

def save_results_to_file(total_returns, sharpe_ratio, file_path='results.txt'):
    with open(file_path, 'w') as file:
        file.write(f"Total Returns: {total_returns}\n")
        file.write(f"Annualized Sharpe Ratio: {sharpe_ratio}\n")

def save_topk_daily_returns(daily_returns, final_values, file_path='topk_daily_returns.txt'):

    with open(file_path, 'w') as f:
        f.write(f"Day 0: Return = {0}, Investment Value = {10000000}\n")
        for i, (r, v) in enumerate(zip(daily_returns, final_values)):
            f.write(f"Day {i + 1}: Return = {r:.2f}, Investment Value = {v:.2f}\n")


if __name__ == '__main__':
    args = parser.parse_args()

    # 确保device是torch.device对象
    if args.device.startswith('cuda') and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device('cpu')

    for i in range(1):
        args = parser.parse_args()
        DEVICE = args.device
        batch_size = args.batch_size
        time_length = args.time_length
        hidden_feat = args.hidden_feat
        A = torch.from_numpy(np.load('/home/xxs/trend/vgnn/First.npy')).float().to(DEVICE)
        n1, n2 = A.shape
        A = A - torch.eye(n1).to(A.device)
        train_loader, val_loader, test_loader = LoadData('/home/xxs/trend/vgnn/daily_indicator.npy', batch_size,
                                                         time_length,
                                                         args.datanormtype, args.task, args.T, args.expect, args.device)

        _, stocks, in_feat = get_dimension(train_loader)
        if args.task == 'trend':
            criterion = nn.CrossEntropyLoss()
        if args.task == 'price':
            criterion = nn.MSELoss()
        if args.task == 'ranking':
            criterion = nn.MSELoss()

        # # 初始化模型和损失函数
        # criterion = nn.CrossEntropyLoss() if args.trend == 'trend' else nn.MSELoss()
        model = MDGNN(args.in_size, args.hidden_feat, num_heads=8, num_layers=2, delta_t=time_length, dropout=0.5)
        model = model.to(device)  # 将模型转移到指定设备上

        # 定义优化器
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_constraint)

        # 训练模型
        train(args, model,train_loader)

        # # 测试和评估
        # test(model, train_loader, A)
        # runtest(model, test_loader, A)
        # daily_evaluate(model, test_loader, A)
        # 计算模型预测
        # predicts, labels, actual_prices = test(model, test_loader, A)
        # daily_returns, final_investment = calculate_daily_returns(predicts, labels, actual_prices)
        # # print('final:',final_investment.shape)
        # daily_sharpe_ratios = [calculate_daily_sharpe_ratio([r]) for r in daily_returns]
        #
        # # print(f"Final Investment Value: {final_investment}")
        # save_daily_results_to_file(final_investment, daily_sharpe_ratios)
        # 获取每日Top-5收益及最终投资
        probs, labels_all, prices_all = test(model, val_loader, A)
        topk_returns, topk_final, topk_daily_values = calculate_topk_returns(probs, labels_all, prices_all, k=5)
        sharpe_topk = calculate_annual_sharpe_ratio_from_returns(topk_returns)

        print(f"Top-5 策略最终投资金额: {topk_final:.2f}")
        print(f"Top-5 年化Sharpe比率: {sharpe_topk:.4f}")

        # 保存每日投资收益
        save_topk_daily_returns(topk_returns, topk_daily_values)

