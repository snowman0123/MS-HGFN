from torch import nn
import torch
from torch.nn import Module, Parameter
import math
import torch.nn.functional as F
import warnings


class Predict(nn.Module):
    def __init__(self, nhi, task):
        super(Predict, self).__init__()
        self.layer1 = nn.Linear(nhi, nhi, bias=True)
        # 使用elif结构确保只有一个分支被执行
        if task == 'price' or task == 'ranking':
            self.pre = nn.Linear(nhi, 1)
        elif task == 'future' or task == 'trend':
            self.pre = nn.Linear(nhi, 2)
        else:
            # 处理无效任务类型，避免self.pred未定义
            raise ValueError(f"Invalid task: {task}")
        self.initialize()

    def initialize(self):
        self.layer1.reset_parameters()
        self.pre.reset_parameters()

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.pre(x).squeeze()
        return x


class linear_attention(nn.Module):
    def __init__(self, node_dim,label_dim,in_dim):
        super(linear_attention,self).__init__()
        self.layerQ = nn.Linear(label_dim, in_dim)
        self.layerK = nn.Linear(node_dim, in_dim)
        self.layerV = nn.Linear(node_dim, in_dim)
        self.initialize()

    def initialize(self):
        self.layerQ.reset_parameters()
        self.layerK.reset_parameters()
        self.layerV.reset_parameters()

    def forward(self, node_emb, future_emb, tau=0.5):
        # pdb.set_trace()
        Q = self.layerQ(future_emb)
        K = self.layerK(node_emb)
        V = self.layerV(node_emb)
        attention_score = torch.matmul(Q, K.transpose(-2, -1))
        attention_weight = F.softmax(attention_score * tau, dim=1)
        z = torch.matmul(attention_weight, V)
        return z


# if __name__ == '__main__':
#     stocks=100
#     node,label=32,64
#     model = linear_attention(node,label,2)
#     x = torch.ones((66, stocks, node))
#     y = torch.ones((66, stocks, label))
#
#     out = model(x,y)
#     print(out.shape)
