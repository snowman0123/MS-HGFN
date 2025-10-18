import torch

import model
from model import *


class Representation(nn.Module):
    def __init__(self,in_features,out_features):
        super(Representation, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.in_features = in_features
        self.out_features = out_features
        #W可训练矩阵
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data)
        #b可训练矩阵
        self.b = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.b.data)
        self.self_attention = Self_attention(in_features, out_features, dropout=0.5, delta_t=5)

    def forward(self, Hvt_delta):
        # 从 Self_attention 模块获取节点表示 zvt
        zvt = self.self_attention(Hvt_delta)
        # 计算股票在交易日 t 获得正收益的概率
        y_probability = self.sigmoid(torch.matmul(zvt, self.W) + self.b)
        return y_probability







