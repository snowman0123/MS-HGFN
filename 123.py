import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# 定义Transformer模型
class StockPredictor(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, num_features):
        super(StockPredictor, self).__init__()

        self.d_model = d_model
        self.embedding = nn.Linear(num_features, d_model)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout),
            num_layers=num_encoder_layers
        )

        self.fc_out = nn.Linear(d_model, 1)  # 输出预测值

    def forward(self, x):
        # x 的形状为 (B, N, T, D)
        B, N, T, D = x.shape

        # 将输入的最后一个维度映射到 d_model 的维度
        x = self.embedding(x)

        # 调整维度以符合 Transformer 的输入要求
        # 将形状从 (B, N, T, D_model) 转换为 (B*N, T, D_model)
        x = x.view(B * N, T, self.d_model)

        # 通过 Transformer 编码器
        x = self.transformer_encoder(x)
        print(x.shape)
        return
        # 对时间维度上的最后一个时间步进行预测（也可以考虑其他聚合方式）
        x = x[:, -1, :]

        # 输出预测结果 (B * N, 1)
        x = self.fc_out(x)

        # 将形状调整回 (B, N, 1)
        x = x.view(B, N, 1)

        return x


# 数据准备（伪造一些股票数据）
B, N, T, D = 16, 5, 20, 10  # Batchsize, 股票数量, 时间步长, 特征维度
data = torch.randn(B, N, T, D)
target = torch.randn(B, N, 1)

# 定义模型和训练参数
d_model = 7
nhead = 1
num_encoder_layers = 3
dim_feedforward = 128
dropout = 0.1

model = StockPredictor(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
                       dim_feedforward=dim_feedforward, dropout=dropout, num_features=D)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型的一个epoch示例
model.train()
for epoch in range(10):
    optimizer.zero_grad()

    # 前向传播
    predictions = model(data)

    # 计算损失
    loss = criterion(predictions, target)

    # 反向传播
    loss.backward()

    # 更新权重
    optimizer.step()

    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# 在此模型的基础上，你可以扩展并调整以适应实际数据集和预测任务。
