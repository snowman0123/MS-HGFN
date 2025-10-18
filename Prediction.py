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


if __name__ == '__main__':
    pass
    # stocks=100
    # node,label=32,64
    # model = linear_attention(node,label,2)
    # x = torch.ones((66, stocks, node))
    # y = torch.ones((66, stocks, label))
    #
    # out = model(x,y)
    # print(out.shape)
