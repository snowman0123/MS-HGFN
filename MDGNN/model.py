from torch import nn
import torch
from torch.nn import Module, Parameter
import math
import torch.nn.functional as F
import warnings
from torch_geometric.nn import GCNConv

warnings.filterwarnings("ignore")


# class GCN(Module):
#     def __init__(self, in_size, hid_size):
#         super(GCN, self).__init__()
#         self.in_features = in_size
#         self.out_features = hid_size
#         self.weight = Parameter(torch.FloatTensor(in_size, hid_size))
#         nn.init.xavier_uniform_(self.weight)
#         self.bias = Parameter(torch.zeros(hid_size))
#
#     def forward(self, x, adj):
#         support = torch.matmul(adj, x)
#         out = torch.matmul(torch.unsqueeze(support, dim=-2), torch.unsqueeze(self.weight, dim=0)).squeeze()
#         return out + self.bias

# 定义DGNN模型
# class DGNN(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(DGNN, self).__init__()
#         self.conv1 = GCNConv(in_channels, 16)
#         self.conv2 = GCNConv(16, out_channels)
#
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x = self.conv1(x, edge_index)
#         x = torch.relu(x)
#         x = self.conv2(x, edge_index)
#         return x

class GCN(Module):
    def __init__(self, in_size, hid_size):
        super(GCN, self).__init__()
        self.in_features = in_size
        self.out_features = hid_size
        self.weight = Parameter(torch.FloatTensor(in_size, hid_size))
        nn.init.xavier_uniform_(self.weight)
        self.bias = Parameter(torch.zeros(hid_size))

    def forward(self, x, adj):
        support = torch.matmul(adj, x)
        out = torch.matmul(torch.unsqueeze(support, dim=-2), torch.unsqueeze(self.weight, dim=0)).squeeze()
        return out + self.bias

class MultiHeadGraphAttention(nn.Module):
    def __init__(self, in_features, out_features, num_heads, dropout):
        super(MultiHeadGraphAttention, self).__init__()
        self.num_heads = num_heads
        self.out_features = out_features

        self.attentions = nn.ModuleList([
            Relation_Attention(in_features, out_features, dropout) for _ in range(num_heads)
        ])
        # 共享矩阵
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

    def forward(self, hi, hj, eij, adj):
        attention_scores = [att(hi, hj, eij) for att in self.attentions]

        Whj = torch.matmul(hj, self.W)
        head_outputs = [torch.matmul(att, Whj) for att in attention_scores]

        # Apply average pooling over the heads
        h_prime = torch.mean(torch.stack(head_outputs), dim=0)
        return F.elu(h_prime)


class Relation_Attention(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(Relation_Attention, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.relu1 = nn.LeakyReLU()
        self.softmax = nn.Softmax()

        # 共享投影矩阵W
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data)

        # 初始化注意力向量 a
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, hi, hj, eij):
        attention_temphi = torch.matmul(self.W, hi)
        attention_temphj = torch.matmul(self.W, hj)
        attention_tempeij = torch.matmul(self.W, eij)
        attention_total = torch.cat(attention_temphi, attention_temphj, attention_tempeij)
        attention = torch.relu(torch.matmul(attention_total, torch.transpose(self.a, -2, -1)))
        attention = self.relu1(attention)
        # 注意力分数
        a = self.softmax(attention)
        # #平均池化
        # s = torch.matmul(a,self.W2,hj)
        a = F.dropout(a, self.dropout, training=self.training)

        return a
        # return torch.mul(Adj, attention)


# 关系感知模块
class Relation_Aware(nn.Module):

    def __init__(self, in_features, out_features, num_heads=1):
        super(Relation_Aware, self).__init__()
        self.mapping=nn.Linear(in_features,5)
        self.in_features = in_features
        self.sigmoid = nn.Sigmoid()
        # 共享投影矩阵W
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data)
        # self.relation_aware=MultiHeadGraphAttention
        self.relation_aware = nn.ModuleList([
            MultiHeadGraphAttention(in_features, out_features, num_heads, .5) for _ in range(num_heads)
        ])
        self.getrelation=nn.Linear(11,1)

    def forward(self, input,adj):
        x=self.mapping(input)
        # print("x",x.shape)
        # print("adj",adj.shape)
        b, n, d = x.shape[0], x.shape[1], x.shape[2]
        x_expanded = x.unsqueeze(1).expand(b, n, n, d)
        x_expanded_transposed = x_expanded.transpose(1, 2)

        adj = adj.unsqueeze(-1).unsqueeze(0).expand(b, n, n, 1)
        relation = torch.cat((x_expanded, x_expanded_transposed,
                         adj), dim=3)
        relation=self.getrelation(relation).squeeze()
        relation=F.softmax(relation,dim=-1)
        embedding=torch.matmul(relation,input)
        embedding=torch.matmul(embedding,self.W)

        return embedding


class MDGNN(nn.Module):
    def __init__(self, in_features, out_features, num_heads, num_layers, delta_t, dropout=0.5):
        super(MDGNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(Relation_Aware(in_features, out_features, num_heads))
        for _ in range(num_layers-1):
            self.layers.append(Relation_Aware(out_features, out_features, num_heads))

        self.delta_t = delta_t
        self.self_attention = Self_attention(out_features, out_features, dropout, delta_t)
    #               B*T*N*D
    def forward(self, x, adj):
        h_list = []
        # print('x.shape',x.shape)
        for i in range(self.delta_t):
            h = x[:, i, :] #B*N*D
            # print('h',h.shape)
            for layer in self.layers:
                h = layer(h, adj)
            h_list.append(h)

        Hvt_delta = torch.stack(h_list, dim=1)  # Shape: (batch_size, delta_t+1, out_features)
        Hvt_delta=torch.transpose(Hvt_delta,1,2)


        m = torch.tril(torch.ones(Hvt_delta.size(1), Hvt_delta.size(1))).unsqueeze(0).to(Hvt_delta.device)

        Z = self.self_attention(Hvt_delta, mask=m)

        return Z


class Self_attention(nn.Module):
    def __init__(self, in_features, out_features, dropout, delta_t):
        super(Self_attention, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.delta_t = delta_t

        self.WQ = nn.Linear(in_features, out_features)
        self.WK = nn.Linear(in_features, out_features)
        self.WV = nn.Linear(in_features, out_features)

        self.softmax = nn.Softmax(dim=-1)

        # 创建相对位置偏置
        self.alibi_bias = self.create_alibi_bias(delta_t)

    def create_alibi_bias(self, seq_len):
        # 根据 ALIBI 的公式，创建一个静态的、不可学习的偏置矩阵
        alibi = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
        alibi = alibi.float().tril()  # 仅考虑过去的位置
        alibi = torch.nn.functional.relu(alibi)  # 保证偏置是非负的
        return alibi

    def forward(self, inputdata, mask=None):
        Q = self.WQ(inputdata)
        K = self.WK(inputdata)
        V = self.WV(inputdata)

        Scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.out_features ** 0.5)
        self.alibi_bias= self.alibi_bias.to(Scores.device)
        # self.alibi_bias=torch.unsqueeze(self.alibi_bias,dim=0)
        # self.alibi_bias = torch.unsqueeze(self.alibi_bias, dim=0)
        # print("Scores shape:", Scores.shape)
        # print("Alibi bias shape:", self.alibi_bias.shape)

        Scores+=self.alibi_bias
        # if mask is not None:
        #     Scores = Scores.masked_fill(mask == 0, float('-inf'))

        Scores_softmax = self.softmax(Scores)
        Scores_softmax = F.dropout(Scores_softmax, self.dropout, training=self.training)
        Market_Signals = torch.matmul(Scores_softmax, V)
        return Market_Signals[:,:,-1,:]


# if __name__ == '__main__':
#     in_feat, out_feat, time_length, stocks = 5, 32, 20, 100
#     model = MDGNN(in_feat, out_feat, 8,2,time_length,dropout=.5)
#     x = torch.ones((66, time_length, stocks, in_feat))  # B T N D
#     out = model(x,torch.eye(100))
#     print(out.shape)
