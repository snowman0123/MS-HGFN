import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class LabelsModule:
    def __init__(self, data, stock_columns, benchmark_column):
        """
        初始化标签模块。

        :param data: 包含股票价格和基准指数数据的DataFrame
        :param stock_columns: 包含股票价格的列名
        :param benchmark_column: 基准指数的列名
        """
        self.data = data.copy()
        self.stock_columns = stock_columns
        self.benchmark_column = benchmark_column

    def compute_returns(self):
        """
        计算股票收益率和基准指数收益率，并生成标签。
        """
        self.data['benchmark_return'] = self.data[self.benchmark_column].pct_change()

        for stock in self.stock_columns:
            self.data[f'{stock}_return'] = self.data[stock].pct_change()
            self.data[f'{stock}_label'] = self.data[f'{stock}_return'] - self.data['benchmark_return']

        self.data.dropna(inplace=True)

    def get_data_with_labels(self):
        """
        获取包含标签的数据。

        :return: 包含标签的数据的DataFrame
        """
        return self.data





class StockDataset(Dataset):
    def __init__(self, data, feature_columns, label_columns):
        """
        初始化股票数据集。

        :param data: 包含特征和标签的数据的DataFrame
        :param feature_columns: 特征列名
        :param label_columns: 标签列名
        """
        self.data = data
        self.feature_columns = feature_columns
        self.label_columns = label_columns

    def __len__(self):
        """
        返回数据集的大小。
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        根据索引返回数据集中的一个样本，包括特征和标签。

        :param idx: 样本索引
        :return: 特征和标签
        """
        features = self.data.iloc[idx][self.feature_columns].values
        labels = self.data.iloc[idx][self.label_columns].values
        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

