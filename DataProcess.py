import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Subset
import os
from sklearn.preprocessing import StandardScaler

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch


class StockDataset(Dataset):
    def __init__(self, stockfile, timelength, normtype, task, device, augmentation=False):
        self.normtype = normtype
        self.stock_data = torch.transpose(torch.from_numpy(np.load(stockfile)), 0, 1).float().to(device)  # [T, N, D]
        self.close = self.stock_data[:, :, 1]
        self.trend = self.process_trend(self.close)
        self.time_length = timelength
        self.task = task
        self.augmentation = augmentation
        if self.normtype == 'meanstd':
            self.stock_data = self.meanstdnorm(self.stock_data)
        if self.normtype == 'maxmin':
            self.stock_data = self.maxminnorm(self.stock_data)

    def maxminnorm(self, stock):
        normedstock = None
        if self.stock_data is not None:
            min_values, _ = torch.min(stock, dim=0)
            max_values, _ = torch.max(stock, dim=0)
            normedstock = (stock - min_values) / (max_values - min_values)
        return normedstock

    def meanstdnorm(self, stock):
        normedstock = None
        if self.stock_data is not None:
            mean_values = torch.mean(stock, dim=0)
            std_values = torch.std(stock, dim=0)
            normedstock = (stock - mean_values) / std_values
        return normedstock

    def process_trend(self, close):
        T, N = close.shape
        result = torch.zeros_like(close).to(close.device)
        for i in range(1, T):
            for j in range(N):
                if close[i, j] > close[i - 1, j]:
                    result[i, j] = 1
                else:
                    result[i, j] = 0
        return result

    # def add_noise(self, data, noise_level=0.01):
    #     noise = torch.randn_like(data) * noise_level
    #     return data + noise

    # def time_shift(self, data, shift_amount):
    #     """平移数据，shift_amount 为平移的步长，可以是正数或负数"""
    #     shifted_data = torch.roll(data, shifts=shift_amount, dims=0)
    #     return shifted_data

    def __len__(self):
        return len(self.stock_data) - self.time_length

    def __getitem__(self, idx):
        label = None

        data = self.stock_data[idx:idx + self.time_length]
        # data['price'] = self.close[idx]

        # if self.augmentation:
        #     shift_amount = np.random.randint(-5, 5)
        #     data = self.time_shift(data, shift_amount)


            # data = self.add_noise(data, noise_level=0.005)
            # 获取实际价格（假设price在时间步长的最后一个索引）
        actual_price = self.close[idx]
        if self.task == 'price':
            label = self.close[idx + self.time_length]
        if self.task == 'ranking':
            label = (self.close[idx + self.time_length] - self.close[idx + self.time_length - 1]) / self.close[
                idx + self.time_length - 1]
        if self.task == 'trend':
            label = self.trend[idx].long()
        # if self.task == 'future':
        #     max_values, _ = torch.max(self.close[idx + self.time_length:idx + self.time_length + self.T], dim=0)
        #     future = (max_values - self.close[idx + self.time_length - 1]) / self.close[idx + self.time_length - 1]
        #     futureres = torch.where(future > self.expect, torch.tensor(1.0), torch.tensor(0.0))
        #     label = futureres.long()

        # data = {}

        # data['price'] = self.close[idx + self.time_length - 1]
        #
        #
        # # data['rate'] = (self.close[idx + self.time_length] - self.close[idx + self.time_length - 1]) / self.close[
        # #     idx + self.time_length - 1]
        # if self.task == 'price':
        #     label = self.close[idx + self.time_length]
        # if self.task == 'ranking':
        #     label = (self.close[idx + self.time_length] - self.close[idx + self.time_length - 1]) / self.close[
        #         idx + self.time_length - 1]
        # if self.task == 'trend':
        #     label = self.future[idx + self.time_length].long()





        #
        # if self.task == 'price':
        #     label = self.close[idx + self.time_length]
        # if self.task == 'ranking':
        #     label = (self.close[idx + self.time_length] - self.close[idx + self.time_length - 1]) / self.close[
        #         idx + self.time_length - 1]
        # if self.task == 'trend':
        #     label = self.trend[idx + self.time_length].long()

        return data, label, actual_price


def LoadData(stockfile, batch_size, time_length, normtype, task, device):
    print('Loading...')
    stock_dataset = StockDataset(stockfile, time_length, normtype, task, device)
    total_samples = len(stock_dataset)
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
