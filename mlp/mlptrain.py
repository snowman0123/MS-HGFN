import copy
import time
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, mean_squared_error, mean_absolute_error
from torch import nn, optim
import os

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

parser.add_argument('--device', type=str, default='cuda:0',
                    help='GPU To Use')

parser.add_argument('--epochs', type=int, default=100,
                    help='Training max epoch')

parser.add_argument('--lr', type=float, default=5e-4,
                    help='Learning rate')

parser.add_argument('--weight_constraint', type=float, default=5e-4,
                    help='L2 Weight Constraint')

parser.add_argument('--time_length', type=int, default=16,
                    help='time length')

parser.add_argument('--hidden_feat', type=int, default=16)

parser.add_argument('--task', type=str, default='ranking')

parser.add_argument('--dropout', type=float, default=.5,)

parser.add_argument('--batch_size', type=int, default=16)

parser.add_argument('--datanormtype', type=str, default='meanstd')

parser.add_argument('--down_sampling_layers', type=int, default=2)

parser.add_argument('--kernel_size', type=int, default=3)

parser.add_argument('--in_size', type=int, default=5)

parser.add_argument('--nhead', type=int, default=1)

parser.add_argument('--attfusion', type=bool, default=True)


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def test(dataloader):
    predicts = None
    labels = None
    early_stopping.best_model.eval()

    for data, label in dataloader:
        out = model(data).reshape(-1)
        out = out.cpu().detach()
        # out = torch.argmax(out.cpu().detach(), dim=-1).reshape(-1)
        label = label.cpu().detach().reshape(-1)
        if predicts == None:
            predicts = out
            labels = label
        else:
            predicts = torch.cat([predicts, out])
            labels = torch.cat([labels, label])
    # print("比例是:{:.2f}".format((torch.sum(torch.eq(predicts, 1)) / len(predicts)).item() * 100), '%', end=' ')
    # print("比例是:{:.2f}".format((torch.sum(torch.eq(labels, 1)) / len(labels)).item() * 100), '%', end=' ')
    print("MSE:{:.4f}".format(mean_squared_error(predicts, labels)),
          'MAE是:{:.4f}'.format(mean_absolute_error(predicts, labels)))
    return mean_squared_error(predicts, labels),mean_absolute_error(predicts, labels)


def train(args, train_loader, val_loader):
    for epoch in range(args.epochs):
        train_losses, val_losses = [], []
        for data, label in tqdm(train_loader):
            model.train()
            optimizer.zero_grad()
            # print(data['indicator'].shape,data['future'].shape,label.shape)
            out = model(data)
            out = out.reshape(-1)
            label = label.reshape(-1)
            # print(out.shape,label.shape)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            with torch.no_grad():
                model.eval()
                for data, label in val_loader:
                    out = model(data)
                    out = out.reshape(-1)
                    label = label.reshape(-1)
                    loss = criterion(out, label)
                    val_losses.append(loss.item())

        print('epoch:{0:}, train_loss:{1:.5f}, val_loss:{2:.5f}'.format(epoch + 1, np.mean(train_losses),
                                                                        np.mean(val_losses)))

        early_stopping(np.mean(val_losses), model,epoch)
        if early_stopping.early_stop:
            print("Early stopping with best_score:{}".format(-early_stopping.best_ls))
            break
        if np.isnan(np.mean(val_losses)) or np.isnan(np.mean(train_losses)):
            break


def mean_and_variance(numbers):
    mean = sum(numbers) / len(numbers)
    variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
    return mean, variance


if __name__ == '__main__':
    MSE = []
    MAE = []

    for i in range(5):
        args = parser.parse_args()
        DEVICE = args.device
        batch_size = args.batch_size
        time_length = args.time_length
        hidden_feat = args.hidden_feat
        train_loader, val_loader, test_loader = LoadData('daily_indicator.npy', batch_size,
                                                         time_length,
                                                         args.datanormtype, args.task, args.device)

        _, stocks, in_feat = get_dimension(train_loader)
        if args.task == 'trend':
            criterion = nn.CrossEntropyLoss()
        if args.task == 'price' or args.task == 'ranking':
            criterion = nn.MSELoss()

        model = Model(args.in_size, args.hidden_feat,args.time_length,args.task)
        model.cuda(device=DEVICE)
        model = model.float()
        early_stopping = EarlyStopping(args, DEVICE, stocks)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_constraint)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
        #                                                        patience=30,
        #                                                        factor=0.1,
        #                                                        verbose=True)

        train(args, train_loader, val_loader)
        mse,mae=test(test_loader)
        MSE.append(mse)
        MAE.append(mae)
    print(np.mean(MSE),np.mean(MAE))
