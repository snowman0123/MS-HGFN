import time
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, mean_squared_error, mean_absolute_error
from torch import nn, optim
import os
from DataProcess import *
from Model import AD_GAT
# from evaluate import evaluate, daily_evaluate1, calculate_sharpe_ratio
from utils import *
import matplotlib.pyplot as plt

torch.set_printoptions(precision=5, sci_mode=False)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from torch import nn, optim

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda:0',
                    help='GPU To Use')

parser.add_argument('--epochs', type=int, default=10,
                    help='Training max epoch')

parser.add_argument('--lr', type=float, default=5e-4,
                    help='Learning rate')

parser.add_argument('--weight_constraint', type=float, default=5e-4,
                    help='L2 Weight Constraint')

parser.add_argument('--time_length', type=int, default=30,
                    help='time length')

parser.add_argument('--hidden_feat', type=int, default=32, )

parser.add_argument('--task', type=str, default='trend', )

parser.add_argument('--alpha', type=float, default=1., )

parser.add_argument('--dropout', type=float, default=.5, )

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
#     model.eval()
#     for data, label in dataloader:
#         out= model(data['indicator'].squeeze())
#         out = torch.argmax(out.cpu().detach(), dim=-1).reshape(-1)
#         label = label.cpu().detach().reshape(-1)
#         if predicts == None:
#             predicts = out
#             labels = label
#         else:
#             predicts = torch.cat([predicts, out])
#             labels = torch.cat([labels, label])
#
#     print("比例是:{:.2f}".format((torch.sum(torch.eq(labels, 1)) / len(labels)).item() * 100), '%', end=' ')
#     print("精度是:{:.2f}".format(accuracy_score(predicts, labels) * 100), '% ',
#           'MCC是:{:.4f}'.format(matthews_corrcoef(predicts, labels)))

def test(model, dataloader):
    all_probs = []
    all_labels = []
    all_prices = []

    model.eval()
    for data, label in dataloader:
        with torch.no_grad():
            out = model(data['indicator'].squeeze())  # shape: [batch, stocks, classes]
            # 如果 out 是二维的，则增加一个批次维度
            if out.dim() == 2:
                out = out.unsqueeze(0)  # 增加批次维度，变为 [1, stocks, classes]
            # print(out.shape)
            probs = F.softmax(out, dim=-1)[..., 1]  # 取每个股票上涨的概率
            all_probs.append(probs.cpu().numpy())
            all_labels.append(label.cpu().numpy())
            if 'price' in data:
                all_prices.append(data['price'].cpu().numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels), np.concatenate(all_prices)



def train(args, model, train_loader):
    for epoch in range(args.epochs):
        avg_loss = 0
        model.train()
        print('epoch ' + str(epoch), end=':')
        for data, label in train_loader:
            label = label.squeeze()
            optimizer.zero_grad()
            out = model(data['indicator'].squeeze())
            # print('out',out.shape)

            _, c = out.shape
            out = out.reshape(-1, c)
            label = label.reshape(-1)
            loss = criterion(out, label)

            avg_loss += loss
            loss.backward()

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
        A = torch.from_numpy(np.load('/home/xxs/trend/vgnn/sp100_graph_relation.npy')).float().to(DEVICE)
        n1, n2 = A.shape
        A = A - torch.eye(n1).to(A.device)
        train_loader, val_loader, test_loader = LoadData('/home/xxs/trend/vgnn/SP100.npy', batch_size,
                                                         time_length,
                                                         args.datanormtype, args.task, args.T, args.expect, args.device)

        _, stocks, in_feat = get_dimension(train_loader)
        if args.task == 'future' or args.task=='trend':
            criterion = nn.CrossEntropyLoss()
        if args.task == 'price':
            criterion = nn.MSELoss()

        model = AD_GAT(96, 5, 60, 6, 60)
        model.cuda(device=DEVICE)
        model = model.float()

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_constraint)
        train(args, model, train_loader)
        # test(model, train_loader, A)
        # test(model, test_loader, A)
        # daily_evaluate(model, test_loader, A)

        probs, labels_all, prices_all = test(model, val_loader)
        topk_returns, topk_final, topk_daily_values = calculate_topk_returns(probs, labels_all, prices_all, k=5)
        sharpe_topk = calculate_annual_sharpe_ratio_from_returns(topk_returns)

        print(f"Top-5 策略最终投资金额: {topk_final:.2f}")
        print(f"Top-5 年化Sharpe比率: {sharpe_topk:.4f}")

        # 保存每日投资收益
        save_topk_daily_returns(topk_returns, topk_daily_values)
