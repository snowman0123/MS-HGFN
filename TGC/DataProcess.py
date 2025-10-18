import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Subset
import os
import torch
import pandas as pd


class StockDataset(Dataset):
    def __init__(self, stockfile, timelength, normtype, task, T, expect, device):  # T是5 10 20 30
        self.T = T
        self.expect = expect
        self.normtype = normtype
        self.stock_data = torch.transpose(torch.from_numpy(np.load(stockfile)), 0, 1).float().to(device)  # [T, N, D]
        self.close = self.stock_data[:, :, 1]
        self.future = self.process_future(self.close)
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

    def process_future(self, close):
        T, N = close.shape
        result = torch.zeros_like(close).to(close.device)
        for i in range(1, T):
            for j in range(N):
                # result[i, j]=close[i, j]/close[i-1,j]
                if close[i, j] > close[i - 1, j]:
                    result[i, j] = 1
        return result

    def meanstdnorm(self, stock):
        normedstock = None
        if self.stock_data != None:
            mean_values = torch.mean(stock, dim=0)
            std_values = torch.std(stock, dim=0)
            normedstock = (stock - mean_values) / std_values
        return normedstock

    def __len__(self):
        return len(self.stock_data) - self.time_length - self.T

    def __getitem__(self, idx):
        label = None
        data = {}
        data['indicator'] = self.stock_data[idx:idx + self.time_length]
        #print("data['indicator'].shape",data['indicator'].shape)
        data['future'] = self.future[idx + self.time_length:idx + self.time_length + self.T]
        data['price'] = self.close[idx + self.time_length - 1]
        data['futurerate'] = (self.close[idx + self.time_length:idx + self.time_length + self.T] - self.close[
            idx + self.time_length - 1]) / \
                             self.close[idx + self.time_length - 1]

        # data['rate'] = (self.close[idx + self.time_length] - self.close[idx + self.time_length - 1]) / self.close[
        #     idx + self.time_length - 1]
        if self.task == 'price':
            label = self.close[idx + self.time_length]
        if self.task == 'ranking':
            label = (self.close[idx + self.time_length] - self.close[idx + self.time_length - 1]) / self.close[
                idx + self.time_length - 1]
        if self.task == 'trend':
            label = self.future[idx + self.time_length].long()
        if self.task == 'future':
            max_values, _ = torch.max(self.close[idx + self.time_length:idx + self.time_length + self.T], dim=0)
            future = (max_values - self.close[idx + self.time_length - 1]) / self.close[idx + self.time_length - 1]
            futureres = torch.where(future > self.expect, torch.tensor(1.0), torch.tensor(0.0))
            label = futureres.long()

        return data, label


def LoadData(stockfile, batch_size, time_length, normtype, task, T, expect, device):
    print('Loading...')
    stock_dataset = StockDataset(stockfile, time_length, normtype, task, T, expect, device)
    total_samples = len(stock_dataset)
    print(total_samples)
    train_size = int(0.75 * total_samples)
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
