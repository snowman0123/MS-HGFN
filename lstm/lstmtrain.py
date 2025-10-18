import time
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, mean_squared_error, mean_absolute_error
from torch import nn, optim
import os

from tqdm import tqdm

from DataProcess import *
from evaluate import daily_evaluate1, evaluate
from model import Model

from utils import *
import matplotlib.pyplot as plt
from LSTM import *

torch.set_printoptions(precision=5, sci_mode=False)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from torch import nn, optim

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda:0',
                    help='GPU To Use')

parser.add_argument('--epochs', type=int, default=100,
                    help='Training max epoch')

parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate')

parser.add_argument('--weight_constraint', type=float, default=5e-4,
                    help='L2 Weight Constraint')

parser.add_argument('--time_length', type=int, default=1,
                    help='time length')

parser.add_argument('--gcnhidden_feat', type=int, default=32,
                    help='gcnhidden_feat')

parser.add_argument('--hidden_feat', type=int, default=32, )

parser.add_argument('--task', type=str, default='trend', )

parser.add_argument('--alpha', type=float, default=1., )

parser.add_argument('--dropout', type=float, default=.5, )

parser.add_argument('--batch_size', type=int, default=64)

parser.add_argument('--datanormtype', type=str, default='meanstd')

parser.add_argument('--in_size', type=int, default=5)

parser.add_argument('--T', type=int, default=1)

parser.add_argument('--expect', type=float, default=0.00)

parser.add_argument('--gnnmodel', type=str, default='gcn')


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def test(model, dataloader):
    all_probs = []
    all_labels = []
    all_prices = []

    model.eval()
    for data, label in dataloader:
        with torch.no_grad():
            out = model(data['indicator'])  # shape: [batch, stocks, classes]
            probs = F.softmax(out, dim=-1)[..., 1]  # 取每个股票上涨的概率
            all_probs.append(probs.cpu().numpy())
            all_labels.append(label.cpu().numpy())
            if 'price' in data:
                all_prices.append(data['price'].cpu().numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels), np.concatenate(all_prices)


        # 如果模型输出是 [B, N], 添加一维以兼容 softmax
        # 如果模型输出是 [B, N], 添加一维以兼容 softmax

    #     # Convert predictions to single class labels
    #     out = torch.argmax(out, dim=-1)
    #     probs = F.softmax(out, dim=-1)  # 转为概率
    #     preds = torch.argmax(probs, dim=-1)  # [B, N]
    #
    #     # 获取每个样本中 top-1 股票索引（按上涨类别的概率）
    #     top1_idx = torch.argmax(probs[:, :, 1], dim=1)  # 每个样本中，对应第1类的最大概率股票
    #
    #     for i, idx in enumerate(top1_idx):
    #         top1_predictions.append(preds[i, idx].item())
    #         top1_labels.append(label[i, idx].item())
    #         if 'price' in data:
    #             top1_prices.append(data['price'][i][idx].item())
    #
    # print("Top-1 精度: {:.2f}%".format(accuracy_score(top1_predictions, top1_labels) * 100))
    # print("Top-1 MCC: {:.4f}".format(matthews_corrcoef(top1_predictions, top1_labels)))
    #
    # return top1_predictions, top1_labels, top1_prices

    #     # If 'label' is multi-dimensional, select single target label
    #     if label.dim() > 1:
    #         label = torch.argmax(label, dim=1)
    #     if 'price' in data:
    #         actual_prices.extend(data['price'].cpu().detach().numpy())
    #
    #     predicts.append(out.cpu().detach())
    #     labels.append(label.cpu().detach())
    #
    # predicts = torch.cat(predicts)
    # labels = torch.cat(labels)
    #
    # print(f"Final shape: predicts {predicts.shape}, labels {labels.shape}")
    #
    # if predicts.size(0) != labels.size(0):
    #     raise ValueError("predicts 和 labels 的样本数量不匹配")
    #
    # print("比例是:{:.2f}".format((torch.sum(torch.eq(labels, 1)) / len(labels)).item() * 100), '%', end=' ')
    # print("精度是:{:.2f}".format(accuracy_score(predicts, labels) * 100), '% ',
    #       'MCC是:{:.4f}'.format(matthews_corrcoef(predicts, labels)))
    # # 返回预测、标签和实际价格
    # return predicts.numpy(), labels.numpy(), actual_prices  # 转换为numpy以便于后续处理


def runtrain(args,model, train_loader, val_loader):
    for epoch in range(args.epochs):
        avg_loss = 0
        train_losses, val_losses = [], []
        for data, label in tqdm(train_loader):
            model.train()
            optimizer.zero_grad()
            # print(data['indicator'].shape,data['future'].shape,label.shape)
            out = model(data['indicator'])
            _, _, c = out.shape
            out = out.reshape(-1, c)
            label = label.reshape(-1)
            loss = criterion(out, label)

            avg_loss += loss
            loss.backward()
            optimizer.step()

        avg_loss = avg_loss / len(train_loader)
        print(avg_loss)
        #     out = out.reshape(-1)
        #     label = label.reshape(-1)
        #     # print(out.shape,label.shape)
        #     loss = criterion(out, label)
        #     loss.backward()
        #     optimizer.step()
        #     train_losses.append(loss.item())
        #
        #     with torch.no_grad():
        #         model.eval()
        #         for data, label in val_loader:
        #             out = model(data['indicator'])
        #             out = out.reshape(-1)
        #             label = label.reshape(-1)
        #             loss = criterion(out, label)
        #             val_losses.append(loss.item())
        #
        # print('epoch:{0:}, train_loss:{1:.5f}, val_loss:{2:.5f}'.format(epoch + 1, np.mean(train_losses),
        #                                                                 np.mean(val_losses)))
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



def calculate_annual_sharpe_ratio(returns, risk_free_rate=0):
    if len(returns) == 0 or np.std(returns) == 0:
        return np.nan  # 或者选择合适的默认值
    mean_return = np.mean(returns)
    std_dev = np.std(returns)

    trading_days = 252  # 假设一年252个交易日
    annual_sharpe_ratio = ((mean_return - risk_free_rate) / std_dev) * np.sqrt(trading_days)

    return annual_sharpe_ratio
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


def calculate_annual_sharpe_ratio_from_returns(returns, risk_free_rate=0):
    returns = np.array(returns)
    if len(returns) == 0 or np.std(returns) == 0:
        return np.nan
    daily_mean = np.mean(returns)
    daily_std = np.std(returns)
    sharpe_ratio = (daily_mean - risk_free_rate) / daily_std * np.sqrt(252)
    return sharpe_ratio


def save_topk_daily_returns(daily_returns, final_values, file_path='topk_daily_returns.txt'):

    with open(file_path, 'w') as f:
        f.write(f"Day 0: Return = {0}, Investment Value = {10000000}\n")
        for i, (r, v) in enumerate(zip(daily_returns, final_values)):
            f.write(f"Day {i + 1}: Return = {r:.2f}, Investment Value = {v:.2f}\n")

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




if __name__ == '__main__':
    # set_random_seed(1)

    args = parser.parse_args()
    DEVICE = args.device
    batch_size = args.batch_size
    time_length = args.time_length
    hidden_feat = args.hidden_feat
    train_loader, val_loader, test_loader = LoadData('/home/xxs/trend/vgnn/CSI500.npy', batch_size,
                                                     time_length,
                                                     args.datanormtype, args.task, args.T,args.expect, args.device)

    _, stocks, in_feat = get_dimension(train_loader)
    if args.task == 'future':
        criterion = nn.CrossEntropyLoss()
    if args.task == 'ranking':
        criterion = nn.MSELoss()
    if args.task == 'trend':
        criterion = nn.CrossEntropyLoss()

    model = Model(args.in_size, hidden_feat, time_length, stocks, args.task)
    model.cuda(device=DEVICE)
    model = model.float()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_constraint)

    runtrain(args, model, train_loader,val_loader)

    # # runtest(model,test_loader)
    # predicts, labels, actual_prices = runtest(model, test_loader)
    # daily_returns, final_investment = calculate_daily_returns(predicts, labels, actual_prices)
    # daily_sharpe_ratios = [calculate_daily_sharpe_ratio([r]) for r in daily_returns]
    #
    # # print(f"Final Investment Value: {final_investment}")
    # save_daily_results_to_file(final_investment, daily_sharpe_ratios)
    probs, labels_all, prices_all = test(model, test_loader)
    topk_returns, topk_final, topk_daily_values = calculate_topk_returns(probs, labels_all, prices_all, k=5)
    sharpe_topk = calculate_annual_sharpe_ratio_from_returns(topk_returns)

    print(f"Top-5 策略最终投资金额: {topk_final:.2f}")
    print(f"Top-5 年化Sharpe比率: {sharpe_topk:.4f}")

    # 保存每日投资收益
    save_topk_daily_returns(topk_returns, topk_daily_values)
