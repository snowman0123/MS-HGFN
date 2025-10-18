import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Subset
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch


class StockDataset(Dataset):
    def __init__(self, stockfile, timelength, normtype, task, device):
        self.normtype = normtype
        self.stock_data = torch.transpose(torch.from_numpy(np.load(stockfile)), 0, 1).float().to(device)  # [T, N, D]
        self.close = self.stock_data[:, :, 1]
        self.trend = self.process_trend(self.close)
        self.time_length = timelength
        self.task = task
        if self.normtype == 'meanstd':
            self.stock_data = self.meanstdnorm(self.stock_data)
        if self.normtype == 'maxmin':
            self.stock_data = self.maxminnorm(self.stock_data)

    def maxminnorm(self, stock):
        normedstock = None
        if self.stock_data != None:
            min_values, _ = torch.min(stock, dim=0)
            max_values, _ = torch.max(stock, dim=0)
            normedstock = (stock - min_values) / (max_values - min_values)
        return normedstock

    def process_trend(self, close):
        T, N = close.shape
        result = torch.zeros_like(close).to(close.device)
        for i in range(1, T):
            for j in range(N):
                if close[i, j] > close[i - 1, j]:
                    result[i, j] = 1
                # if (close[i, j] - close[i-1, j]) / close[i-1, j] * 100 > .5:
                #     result[i, j] = 2
                # elif (close[i, j] - close[i-1, j]) / close[i-1, j] * 100 < -.5:
                #     result[i, j] = 0
                # else:
                #     result[i, j] = 1

        return result

    def meanstdnorm(self, stock):
        normedstock = None
        if self.stock_data != None:
            mean_values = torch.mean(stock, dim=0)
            std_values = torch.std(stock, dim=0)
            normedstock = (stock - mean_values) / std_values
        return normedstock

    def __len__(self):
        return len(self.stock_data) - self.time_length

    def __getitem__(self, idx):
        label = None
        data = self.stock_data[idx:idx + self.time_length]

        if self.task == 'price':
            label = self.close[idx + self.time_length]
        if self.task == 'ranking':
            label = (self.close[idx + self.time_length] - self.close[idx + self.time_length - 1]) / self.close[
                idx + self.time_length - 1]
        if self.task == 'trend':
            label = self.trend[idx + self.time_length].long()

        return data, label


def LoadData(stockfile, batch_size, time_length, normtype, task, device):
    print('Loading...')
    stock_dataset = StockDataset(stockfile, time_length, normtype, task, device)
    total_samples = len(stock_dataset)
    train_size = int(0.8 * total_samples)
    val_test_size = total_samples - train_size
    val_size = int(0.5 * val_test_size)

    train_set = Subset(stock_dataset, range(train_size - time_length))
    val_set = Subset(stock_dataset, range(train_size - time_length, train_size + val_size - time_length))
    test_set = Subset(stock_dataset, range(train_size + val_size - time_length, total_samples - time_length))

    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)

    # print('Done!')
    return train_loader, val_loader, test_loader
