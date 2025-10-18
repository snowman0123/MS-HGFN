import copy
import time
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, matthews_corrcoef, recall_score, f1_score, mean_squared_error, mean_absolute_error, roc_auc_score
from torch import nn, optim
import os
from sklearn.model_selection import KFold, StratifiedKFold
from collections import Counter
from tqdm import tqdm

from DataProcess import *
from Teacher_Student.model import Teacher, Student
from evaluate import evaluate
from model import Model, EarlyStopping
from utils import *
import matplotlib.pyplot as plt
from LSTM import *

torch.set_printoptions(precision=5, sci_mode=False)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from torch import nn, optim

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda',
                    help='Device to use for computation')

parser.add_argument('--epochs', type=int, default=20,
                    help='Training max epoch')

parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate')

parser.add_argument('--weight_constraint', type=float, default=0,
                    help='L2 Weight Constraint')

parser.add_argument('--time_length', type=int, default=16,
                    help='time length')

parser.add_argument('--hidden_feat', type=int, default=16)

parser.add_argument('--task', type=str, default='trend')

parser.add_argument('--dropout', type=float, default=.5)

parser.add_argument('--batch_size', type=int, default=1)

parser.add_argument('--datanormtype', type=str, default='meanstd')

parser.add_argument('--down_sampling_layers', type=int, default=2)

parser.add_argument('--num_layers', type=int, default=2)

parser.add_argument('--num', type=int, default=2)

parser.add_argument('--kernel_size', type=int, default=3)

parser.add_argument('--in_size', type=int, default=5)

parser.add_argument('--nhead', type=int, default=1)

parser.add_argument('--attfusion', type=bool, default=True)

parser.add_argument('--d_model', type=int, default=16)

# parser.add_argument('--d_ff', type=int, default=16)

# parser.add_argument('--dmodel', type=int, default=16)

parser.add_argument('--output_attention', type=bool, default=False)

parser.add_argument('--activation', type=str, default='relu')

parser.add_argument('--embed', type=str, default='fixed')

parser.add_argument('--freq', type=str, default='t', help='时间特征的频率')

parser.add_argument('--factor', type=int, default=5, help='注意力的缩放因子')

parser.add_argument('--hidden_size', type=int, default=5)

parser.add_argument('--max_len', type=int, default=16)

parser.add_argument('--d_ff', type=int, default=2, help='dimension of fcn')

parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')

parser.add_argument('--time_step', type=int, default=8)

parser.add_argument('--augmentation', type=bool, default=True)

parser.add_argument('--T', type=int, default=1)

parser.add_argument('--expect', type=float, default=0.04, help='0.04, 0.025, 0.015')





def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def runtest(dataloader):
    predicts = None
    labels = None
    all_probs = None
    # adj_matrices = []  # 初始化邻接矩阵列表
    early_stopping.best_model.eval()

    for data, label,actually_price in dataloader:
        data, label = data.to('cuda:0'), label.to('cuda:0')
        out = model(data.permute(0, 2, 3, 1).to('cuda:0'))

        out = out.squeeze(0)
        # 保存完整的概率分布以计算AUC
        probabilities = torch.softmax(out, dim=-1)
        class_probs = probabilities[:, 1]  # 基于阳性类提取概率，假设阳性类是1

        # 选择最大值的索引（选择类别）
        out = torch.argmax(out, dim=-1)  # 结果为 [196]

        # 将标签整平到一维
        label = label.reshape(-1)
        if predicts == None:
            predicts = out
            labels = label
            all_probs = class_probs
        else:
            predicts = torch.cat([predicts, out])
            labels = torch.cat([labels, label])
            all_probs = torch.cat([all_probs, class_probs])

    predicts = predicts.cpu().numpy() if predicts.is_cuda else predicts.numpy()
    labels = labels.cpu().numpy() if labels.is_cuda else labels.numpy()
    all_probs = all_probs.detach().cpu().numpy()  # 使用 detach() 以分离梯度计算图

    # 计算AUC值
    try:
        auc = roc_auc_score(labels, all_probs) * 100  # AUC 换算为百分比表示
    except ValueError as e:
        auc = None
        print(f"计算AUC时遇到错误: {e}")

    # true_positives = np.sum((predicts == 1) & (predicts == labels))
    #
    # # 计算预测正确的负样本个数
    # true_negatives = np.sum((predicts == 0) & (predicts == labels))
    #
    # print(f"预测正确的正样本个数: {true_positives}")
    # print(f"预测正确的负样本个数: {true_negatives}")
    #
    # unique_labels, counts_labels = np.unique(labels, return_counts=True)
    # print("真实样本分布:")
    # for label, count in zip(unique_labels, counts_labels):
    #     print(f"标签 {label}: {count} 个")
    #
    # # 输出预测样本分布
    # unique_predicts, counts_predicts = np.unique(predicts, return_counts=True)
    # print("预测样本分布:")
    # for label, count in zip(unique_predicts, counts_predicts):
    #     print(f"标签 {label}: {count} 个")
    positive_ratio = np.sum(labels == 1) / len(labels) * 100
    print(f"比例是: {positive_ratio:.2f}%", end=' ')

    # print("比例是:{:.2f}".format((torch.sum(torch.eq(labels, 1)) / len(labels)).item() * 100), '%', end=' ')
    print("精度是:{:.2f}".format(accuracy_score(predicts, labels) * 100), '% ',
          'MCC是:{:.4f}'.format(matthews_corrcoef(predicts, labels)))

# def precision_at_k(predicts, labels, k=5):
#     # 将预测值和真实值进行排序
#     _, top_k_indices = torch.topk(predicts, k)  # 获取前 K 名股票的索引
#     # 获取前 K 名股票的真实标签
#     top_k_labels = labels[top_k_indices]
#
#     # 假设标签为1表示表现良好的股票
#     relevant_count = torch.sum(top_k_labels > 0).item()  # 获取表现良好的股票数
#
#     # 计算 P@K
#     p_at_k = relevant_count / k
#     return p_at_k


# def runtest(dataloader):
#     predicts = None
#     labels = None
#     all_probs = None
#     actual_prices = []  # 添加一个列表来存储实际价格
#     # adj_matrices = []  # 初始化邻接矩阵列表
#     all_probs = []
#     all_labels = []
#     all_prices = []
#
#     early_stopping.best_model.eval()
#
#     for data, label, actual_price in dataloader:
#         data, label = data.to('cuda:0'), label.to('cuda:0')
#         out = model(data.permute(0, 2, 3, 1).to('cuda:0'))
#
#         # out = out.squeeze(0)
#         # 保存完整的概率分布以计算AUC
#         probabilities = torch.softmax(out, dim=-1)
#         class_probs = probabilities[:, 1]  # 基于阳性类提取概率，假设阳性类是1
#         probs = F.softmax(out, dim=-1)[..., 1]  # 取每个股票上涨的概率
#
#         all_probs.append(probs.detach().cpu().numpy())
#         all_labels.append(label.cpu().numpy())
#
#         # if 'price' in data:
#         all_prices.append(actual_price.cpu().detach().numpy())
#
#
#     return np.concatenate(all_probs), np.concatenate(all_labels), np.concatenate(all_prices)
    # return np.concatenate(all_probs), np.concatenate(all_labels), np.concatenate(all_prices)
#

# def runtest(dataloader):
#     predicts = None
#     labels = None
#     all_probs = None
#     # adj_matrices = []  # 初始化邻接矩阵列表
#     early_stopping.best_model.eval()
#
#     for data, label,actual_price in dataloader:
#         data, label = data.to('cuda:0'), label.to('cuda:0')
#         out = model(data.permute(0, 2, 3, 1).to('cuda:0'))
#
#         out = out.squeeze(0)
#         # 保存完整的概率分布以计算AUC
#         probabilities = torch.softmax(out, dim=-1)
#         class_probs = probabilities[:, 1]  # 基于阳性类提取概率，假设阳性类是1
#
#         # 选择最大值的索引（选择类别）
#         out = torch.argmax(out, dim=-1)  # 结果为 [196]
#
#         # 将标签整平到一维
#         label = label.reshape(-1)
#         if predicts == None:
#             predicts = out
#             labels = label
#             all_probs = class_probs
#         else:
#             predicts = torch.cat([predicts, out])
#             labels = torch.cat([labels, label])
#             all_probs = torch.cat([all_probs, class_probs])
#
#     predicts = predicts.cpu().numpy() if predicts.is_cuda else predicts.numpy()
#     labels = labels.cpu().numpy() if labels.is_cuda else labels.numpy()
#     all_probs = all_probs.detach().cpu().numpy()  # 使用 detach() 以分离梯度计算图
#
#     # 计算AUC值
#     try:
#         auc = roc_auc_score(labels, all_probs) * 100  # AUC 换算为百分比表示
#     except ValueError as e:
#         auc = None
#         print(f"计算AUC时遇到错误: {e}")
#
#
#     positive_ratio = np.sum(labels == 1) / len(labels) * 100
#     print(f"比例是: {positive_ratio:.2f}%", end=' ')
#
#     print("比例是:{:.2f}".format((torch.sum(torch.eq(labels, 1)) / len(labels)).item() * 100), '%', end=' ')
#     print("精度是:{:.2f}".format(accuracy_score(predicts, labels) * 100), '% ',
#            'MCC是:{:.4f}'.format(matthews_corrcoef(predicts, labels)))









def runtrain(args, train_loader, val_loader):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    early_stopping = EarlyStopping(args, device, stocks, patience=5)  # 确保已正确初始化

    for epoch in range(args.epochs):
        # === 训练阶段 ===
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for data, labels, _ in tqdm(train_loader):  # 规范命名
            data = data.permute(0, 2, 3, 1).to(device)
            labels = labels.squeeze().long().to(device)

            optimizer.zero_grad()
            outputs = model(data)

            # 维度校验保护
            if outputs.dim() > 2:
                outputs = outputs.view(-1, outputs.size(-1))
            labels = labels.view(-1)

            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # 累计指标
            train_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        # === 验证阶段 ===
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for data, labels, _ in val_loader:
                data = data.permute(0, 2, 3, 1).to(device)
                labels = labels.squeeze().long().to(device)

                outputs = model(data)
                if outputs.dim() > 2:
                    outputs = outputs.view(-1, outputs.size(-1))
                labels = labels.view(-1)

                val_loss += criterion(outputs, labels).item() * data.size(0)

        # === 指标计算 ===
        train_loss = train_loss / total
        val_loss = val_loss / len(val_loader.dataset)
        train_acc = correct / total

        # === 早停机制 ===
        early_stopping(val_loss, model,epoch)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}")
            # model.load_state_dict(torch.load('checkpoint.pt'))  # 恢复最佳参数

            break





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



def calculate_annual_sharpe_ratio_from_returns(returns, risk_free_rate=0):
    returns = np.array(returns)
    if len(returns) == 0 or np.std(returns) == 0:
        return np.nan
    daily_mean = np.mean(returns)
    daily_std = np.std(returns)
    sharpe_ratio = (daily_mean - risk_free_rate) / daily_std * np.sqrt(252)
    return sharpe_ratio








def save_topk_daily_returns(daily_returns, final_values, file_path='topk_daily_returns_500.txt'):

    with open(file_path, 'w') as f:
        f.write(f"Day 0: Return = {0}, Investment Value = {10000000}\n")
        for i, (r, v) in enumerate(zip(daily_returns, final_values)):
            f.write(f"Day {i + 1}: Return = {r:.2f}, Investment Value = {v:.2f}\n")



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
    set_random_seed(3407)
    MSE = []
    MAE = []
    ROI_list = []
    Sharpe_list = []
    P_ak_list = []


    class FocalLoss(nn.Module):
        def __init__(self, alpha=1, gamma=2, reduction='mean'):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction

        def forward(self, inputs, targets):
            CE_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-CE_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * CE_loss

            return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()


    for i in range(1):
        # for dropout in [0.3,0.5]:
        #     for lr in [1e-4,1e-5]:
        args = parser.parse_args()
        DEVICE = args.device
        batch_size = args.batch_size
        time_length = args.time_length
        hidden_feat = args.hidden_feat
    #     # # # #####
    #     data = np.load('/home/xxs/trend/SP100.npy')
    #
    #     # 设置随机种子以确保结果可重复
    #     np.random.seed(3407)
    # #
    #     # 定义样本大小，这里假设选取10%的数据
    #     sample_size = int(0.05 * data.shape[0])
    #
    #     # 随机选择索引
    #     random_indices = np.random.choice(data.shape[0], sample_size, replace=False)
    #
    #     # 提取随机样本
    #     sample_data = data[random_indices, :, :]
    #
    #     # 保存为新的 .npy 文件
    #     np.save('/home/xxs/trend/SP100_sample.npy', sample_data)

        train_loader, val_loader, test_loader = LoadData('/home/xxs/trend/CSI500.npy', batch_size,
                                                         time_length,
                                                         args.datanormtype, args.task, args.device)
        # train_loader, val_loader, test_loader = LoadData('/home/xxs/trend/SP100_sample.npy', batch_size,
        #                                                  time_length,
        #                                                  args.datanormtype, args.task, args.device, args.augmentation)
        # print('train_loader.shape',get_dimension(train_loader))#[16,196,5]
        # from sklearn.utils.class_weight import compute_class_weight
        # import numpy as np
        #
        # all_labels = np.concatenate([labels.cpu().numpy() for _, labels, _ in train_loader])
        # class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(all_labels), y=all_labels)
        # class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)


        def calculate_class_weights(train_loader):
            # 统计训练集中每个类别的样本数
            label_counts = Counter()
            for _, labels, _ in train_loader:
                # 累积计数器中各类别样本数
                label_counts.update(labels.flatten().tolist())

            # 计算每个类别的权重
            total_samples = sum(label_counts.values())
            num_classes = len(label_counts)
            weights = np.zeros(num_classes)

            for label, count in label_counts.items():
                weights[label] = total_samples / (num_classes * count)

            # 转化为 tensor 传入损失函数
            return torch.tensor(weights, dtype=torch.float32).to(args.device)


        # 在定义损失函数之前，计算权重
        class_weights = calculate_class_weights(train_loader)
        # criterion = nn.CrossEntropyLoss(weight=class_weights)

        _, stocks, in_feat = get_dimension(train_loader)
        if args.task == 'trend':
            num_classes = 2  # 设置为类别数量
            # criterion = nn.CrossEntropyLoss(weight=class_weights)
            criterion = FocalLoss(alpha=class_weights[1], gamma=2)

        if args.task == 'price' or args.task == 'ranking':
            num_classes = 1
            criterion = nn.MSELoss()


        device = "cuda:0"


        model = Model(args, stocks, device)

        model.cuda()
        model = model.float()
        early_stopping = EarlyStopping(args, DEVICE, stocks)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_constraint)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
        #                                                        patience=30,
        #                                                        factor=0.1,
        #                                                        verbose=True)

        # 在模型训练完成后调用

        #visualize_adjacency_matrices(model)
        runtrain(args, train_loader, val_loader)
        # runtest(test_loader)
        # predicts, labels, actual_prices = runtest(test_loader)
        # daily_returns, final_investment = calculate_daily_returns(predicts, labels, actual_prices)
        # daily_sharpe_ratios = [calculate_daily_sharpe_ratio([r]) for r in daily_returns]
        #
        # # print(f"Final Investment Value: {final_investment}")
        # save_daily_results_to_file(final_investment, daily_sharpe_ratios)
        probs, labels_all, prices_all = runtest(test_loader)
        topk_returns, topk_final, topk_daily_values = calculate_topk_returns(probs, labels_all, prices_all, k=5)
        sharpe_topk = calculate_annual_sharpe_ratio_from_returns(topk_returns)

        print(f"Top-5 策略最终投资金额: {topk_final:.2f}")
        print(f"Top-5 年化Sharpe比率: {sharpe_topk:.4f}")

        # 保存每日投资收益
        save_topk_daily_returns(topk_returns, topk_daily_values)

    # print(np.mean(MSE),np.mean(MAE),np.mean(ROI_list),np.mean(Sharpe_list),np.mean(P_ak_list))
