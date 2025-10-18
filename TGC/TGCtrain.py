import time
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, mean_squared_error, mean_absolute_error
from torch import nn, optim
import os
from DataProcess import *
from model import TGC
from evaluate import daily_evaluate1
from utils import *
import matplotlib.pyplot as plt
from LSTM import *

torch.set_printoptions(precision=5, sci_mode=False)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from torch import nn, optim

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda:1',
                    help='GPU To Use')

parser.add_argument('--epochs', type=int, default=50,
                    help='Training max epoch')

parser.add_argument('--lr', type=float, default=0.0005,
                    help='Learning rate')

parser.add_argument('--weight_constraint', type=float, default=5e-4,
                    help='L2 Weight Constraint')

parser.add_argument('--time_length', type=int, default=1,
                    help='time length')

parser.add_argument('--gcnhidden_feat', type=int, default=64,
                    help='gcnhidden_feat')

parser.add_argument('--hidden_feat', type=int, default=64, )

parser.add_argument('--task', type=str, default='trend', )

parser.add_argument('--alpha', type=float, default=1., )

parser.add_argument('--dropout', type=float, default=.5, )

parser.add_argument('--batch_size', type=int, default=64)

parser.add_argument('--datanormtype', type=str, default='meanstd')

parser.add_argument('--in_size', type=int, default=5)

parser.add_argument('--T', type=int, default=1)

parser.add_argument('--expect', type=float, default=0.04, help='0.04, 0.025, 0.015')

parser.add_argument('--gnnmodel', type=str, default='GAT')

parser.add_argument('--gnnteachermodel', type=str, default='GCN')


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
#     # print("比例是:{:.2f}".format((torch.sum(torch.eq(labels, 1)) / len(labels)).item() * 100), '%', end=' ')
#     # print("精度是:{:.2f}".format(accuracy_score(predicts, labels) * 100), '% ',
#     #       'MCC是:{:.4f}'.format(matthews_corrcoef(predicts, labels)))
#     #
#     # # 返回预测、标签和实际价格
#     # return predicts.numpy(), labels.numpy(), actual_prices  # 转换为numpy以便于后续处理
#         # 计算每个时间点的前`top_stocks`股票预测的平均值
#     topN_averages = [
#         torch.mean(torch.take(predicts[i], torch.argsort(predicts[i], descending=True)[:top_stocks])).item() for i
#         in range(len(predicts))]
#
#     print("比例是:{:.2f}".format((torch.sum(torch.eq(labels, 1)) / len(labels)).item() * 100), '%', end=' ')
#     print("精度是:{:.2f}".format(accuracy_score(predicts.argmax(dim=-1).numpy(), labels) * 100), '% ',
#               'MCC是:{:.4f}'.format(matthews_corrcoef(predicts.argmax(dim=-1).numpy(), labels)))
#
#     return topN_averages, labels.numpy(), actual_prices

# def test(model, dataloader, A):
#     model.eval()
#     top1_predictions = []
#     top1_labels = []
#     top1_prices = []
#
#
#     for data, label in dataloader:
#         model.eval()
#         out = model(data['indicator'], A)  # [B, N, C]
#         # print('out:',out.shape)
#         probs = F.softmax(out, dim=-1)  # 转为概率
#         preds = torch.argmax(probs, dim=-1)  # [B, N]
#
#         # 获取每个样本中 top-1 股票索引（按上涨类别的概率）
#         top1_idx = torch.argmax(probs[:, :, 1], dim=1)  # 每个样本中，对应第1类的最大概率股票
#
#         for i, idx in enumerate(top1_idx):
#             top1_predictions.append(preds[i, idx].item())
#             top1_labels.append(label[i, idx].item())
#             if 'price' in data:
#                 top1_prices.append(data['price'][i][idx].item())
#
#     print("Top-1 精度: {:.2f}%".format(accuracy_score(top1_predictions, top1_labels) * 100))
#     print("Top-1 MCC: {:.4f}".format(matthews_corrcoef(top1_predictions, top1_labels)))
#
#     return top1_predictions, top1_labels, top1_prices

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

if __name__ == '__main__':
    # set_random_seed(1)
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

        # for data, labels in test_loader:
        #     print('label',labels)
        #     break  # 仅打印第一个批次，您可以省略该行以打印所有批次的标签

        _, stocks, in_feat = get_dimension(train_loader)
        if args.task == 'future':
            criterion = nn.CrossEntropyLoss()
        if args.task == 'price':
            criterion = nn.MSELoss()

        if args.task == 'trend':
            criterion = nn.CrossEntropyLoss()

        smodel = TGC(args.in_size, hidden_feat, time_length, stocks, args.task)
        smodel.cuda(device=DEVICE)
        smodel = smodel.float()

        optimizer = optim.Adam(smodel.parameters(), lr=args.lr, weight_decay=args.weight_constraint)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                               patience=30,
                                                               factor=0.1,
                                                               verbose=True)

        train(args, smodel, train_loader)
        # test(smodel, train_loader, A)
        # # test(smodel, val_loader)
        # test(smodel, test_loader, A)
        # 计算模型预测
        # predicts, labels, actual_prices = test(smodel, test_loader, A)
        # daily_returns, final_investment = calculate_daily_returns(predicts, labels, actual_prices)
        # daily_sharpe_ratios = [calculate_daily_sharpe_ratio([r]) for r in daily_returns]
        #
        # # print(f"Final Investment Value: {final_investment}")
        # save_daily_results_to_file(final_investment, daily_sharpe_ratios)

        # top1_preds, top1_labels, top1_prices = test(smodel, val_loader, A)
        # daily_returns, final_investment = calculate_daily_returns(top1_preds, top1_labels, top1_prices)
        # annual_sharpe = calculate_annual_sharpe_ratio(daily_returns)
        # save_results_to_file(final_investment, annual_sharpe, 'top1_results.txt')
        probs, labels_all, prices_all = test(smodel, test_loader, A)
        topk_returns, topk_final, topk_daily_values = calculate_topk_returns(probs, labels_all, prices_all, k=5)
        sharpe_topk = calculate_annual_sharpe_ratio_from_returns(topk_returns)

        print(f"Top-5 策略最终投资金额: {topk_final:.2f}")
        print(f"Top-5 年化Sharpe比率: {sharpe_topk:.4f}")

        # 保存每日投资收益
        save_topk_daily_returns(topk_returns, topk_daily_values)




