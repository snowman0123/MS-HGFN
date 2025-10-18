import torch
import torch.nn as nn
import math

def compared_version(ver1, ver2):
    """
    :param ver1
    :param ver2
    :return: ver1< = >ver2 False/True
    """
    list1 = str(ver1).split(".")
    list2 = str(ver2).split(".")
    
    for i in range(len(list1)) if len(list1) < len(list2) else range(len(list2)):
        if int(list1[i]) == int(list2[i]):
            pass
        elif int(list1[i]) < int(list2[i]):
            return -1
        else:
            return 1
    
    if len(list1) == len(list2):
        return True
    elif len(list1) < len(list2):
        return False
    else:
        return True

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=196):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, in_size, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if compared_version(torch.__version__, '1.5.0') else 2
        self.tokenConv = nn.Conv1d(in_channels=in_size, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        device = next(self.parameters()).device  # 获取模型的设备
        x = x.to(device)
        # print('xxx11',x.shape)
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        # x = self.tokenConv(x.permute(0, 3, 2, 1))
        return x

# class TokenEmbedding(nn.Module):
#     def __init__(self, in_sizes, d_model):
#         super(TokenEmbedding, self).__init__()
#         self.tokenConvs = nn.ModuleList(
#             [nn.Conv1d(in_channels=in_size, out_channels=d_model, kernel_size=3, padding=1, padding_mode='circular', bias=False)
#              for in_size in in_sizes]
#         )
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
#
#     def forward(self, x):
#         print('xxx11', x.shape)
#
#         # 根据输入通道数选择合适的卷积层
#         in_channels = x.shape[2]
#         for tokenConv in self.tokenConvs:
#             if tokenConv.in_channels == in_channels:
#                 x = tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
#                 break
#         else:
#             # 如果没有找到合适的卷积层，可以考虑抛出异常或返回 None
#             raise ValueError(f"没有找到与输入通道数 {in_channels} 匹配的卷积层")
#
#         return x


# class TokenEmbedding(nn.Module):
#     def __init__(self, in_sizes, d_model):
#         super(TokenEmbedding, self).__init__()
#         self.tokenConvs = nn.ModuleList(
#             [nn.Conv1d(in_channels=in_size, out_channels=d_model, kernel_size=3, padding=1, padding_mode='circular', bias=False)
#              for in_size in in_sizes]
#         )
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
#
#     def forward(self, x):
#         print('xxx11', x.shape)
#
#         # 根据输入通道数选择合适的卷积层
#         in_channels = x.shape[2]
#         for tokenConv in self.tokenConvs:
#             if tokenConv.in_channels == in_channels:
#                 x = tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
#                 break
#
#         return x



class FixedEmbedding(nn.Module):
    def __init__(self, in_size, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(in_size, d_model).float()
        w.require_grad = False

        position = torch.arange(0, in_size).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(in_size, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


# class TemporalEmbedding(nn.Module):
#     def __init__(self, d_model, embed_type='fixed', freq='t'):
#         super(TemporalEmbedding, self).__init__()
#
#         minute_size = 4
#         hour_size = 24
#         weekday_size = 7
#         day_size = 32
#         month_size = 13
#
#         Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
#         if freq == 't':
#             self.minute_embed = Embed(minute_size, d_model)
#         self.hour_embed = Embed(hour_size, d_model)
#         self.weekday_embed = Embed(weekday_size, d_model)
#         self.day_embed = Embed(day_size, d_model)
#         self.month_embed = Embed(month_size, d_model)
# 
#     def forward(self, x):
#         x = x.long()
#
#         minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
#         hour_x = self.hour_embed(x[:, :, 3])
#         weekday_x = self.weekday_embed(x[:, :, 2])
#         day_x = self.day_embed(x[:, :, 1])
#         month_x = self.month_embed(x[:, :, 0])
#
#         return hour_x + weekday_x + day_x + month_x + minute_x


# class TimeFeatureEmbedding(nn.Module):
#     def __init__(self, d_model, embed_type='timeF', freq='h'):
#         super(TimeFeatureEmbedding, self).__init__()
#
#         freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
#         d_inp = freq_map[freq]
#         self.embed = nn.Linear(d_inp, d_model, bias=False)
#
#     def forward(self, x):
#         return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, in_size, d_model, embed_type='fixed', freq='t', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(in_size=in_size, d_model=d_model)
        # self.position_embedding = PositionalEmbedding(d_model=d_model)
        # self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
        #                                             freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
        #     d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # print('x.shape:',x.shape)
        # x = self.value_embedding(x) + self.position_embedding(x)
        x_emb = self.value_embedding(x)  # 先执行卷积操作
        # print('x_emb.shape:', x_emb.shape)  # 打印卷积后的形状
        # pos_emb = self.position_embedding(x_emb)  # 确保位置嵌入与卷积后的长度匹配
        # print('pos_emb.shape:', x_emb.shape)  # 打印卷积后的形状
        x = x_emb
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, in_size, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(in_size=in_size, d_model=d_model)
        # self.position_embedding = PositionalEmbedding(d_model=d_model)
        # self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
        #                                             freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
        #     d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x)
        return self.dropout(x)
