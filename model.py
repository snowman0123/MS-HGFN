import numpy as np
from torch import nn
import torch
from torch.nn import Module, Parameter
import math
import torch.nn.functional as F
import warnings
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
# from mamba_ssm.modules.mamba_simple import Mamba  # 引入Mamba模块
from Transformer import TransformerModel
from GCN import GCNLayer
import os

warnings.filterwarnings("ignore")


class EarlyStopping:
    def __init__(self, args, device, stocks, patience=5, verbose=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_ls = None
        self.best_model = Model(args, stocks, device).to('cuda')
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.trace_func = print

    def __call__(self, val_loss, model, epoch=0):
        ls = val_loss
        if self.best_ls is None:
            self.best_ls = ls
            self.best_model.load_state_dict(model.state_dict())
        elif ls > self.best_ls:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience} with {epoch}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_ls = ls
            self.best_model.load_state_dict(model.state_dict())
            self.counter = 0

# class EarlyStopping:
#     def __init__(self, args, device, stocks, patience=5, verbose=True):
#         self.patience = patience
#         self.verbose = verbose
#         self.counter = 0
#         self.best_ls = None
#         self.best_model = Model(args, stocks, device).to('cuda')
#         self.early_stop = False
#         self.val_loss_min = np.Inf
#         self.trace_func = print
#
#     def __call__(self, val_loss, model, epoch=0):
#         ls = val_loss
#         if self.best_ls is None:
#             self.best_ls = ls
#             self.best_model.load_state_dict(model.state_dict())
#             torch.save(model.state_dict(), 'checkpoint.pt')  # 保存最佳模型
#         elif ls > self.best_ls:
#             self.counter += 1
#             if self.verbose:
#                 self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience} at epoch {epoch}')
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_ls = ls
#             self.best_model.load_state_dict(model.state_dict())
#             torch.save(model.state_dict(), 'checkpoint.pt')  # 更新保存的最佳模型
#             self.counter = 0
# class EarlyStopping:
#     def __init__(self, args, device, stocks, patience=5, verbose=True):
#         self.patience = patience
#         self.verbose = verbose
#         self.counter = 0
#         self.best_model_state = None
#         self.early_stop = False
#         self.val_loss_min = np.Inf
#         self.trace_func = print
#
#     def __call__(self, val_loss, model, epoch=0):
#         if self.best_model_state is None or val_loss < self.val_loss_min:
#             self.best_model_state = model.state_dict()
#             torch.save(self.best_model_state, 'checkpoint.pt')  # 保存最佳模型
#             self.val_loss_min = val_loss
#             self.counter = 0
#         else:
#             self.counter += 1
#             if self.verbose:
#                 self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience} at epoch {epoch}')
#             if self.counter >= self.patience:
#                 self.early_stop = True

class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5, device='cuda'):
        super(GraphConvolutionLayer, self).__init__()
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()
        self.dropout = nn.Dropout(p=dropout)
        # self.norm = nn.LayerNorm(out_features)  # 层归一化更适合序列数据
        # self.activation = nn.GELU()  # 改用GELU激活函数

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        # Step 1: Linear transformation using the learned weights
        support = torch.matmul(x.to(self.device), self.weight.to(self.device))  # (B, N, out_features)

        # Step 2: Aggregate features from the neighborhood using the adjacency matrix
        output = torch.matmul(adj, support)  # (B, N, out_features)
        # output = self.norm(output)  # 直接在输出维度归一化
        # output = self.activation(output)

        # Step 3: Apply dropout
        output = self.dropout(output)

        return output








########################
class RGA_Module(nn.Module):
    def __init__(self, in_channel, in_spatial, cha_ratio=1, spa_ratio=1, h_dim=16):
        super(RGA_Module, self).__init__()

        # self.num_stocks = num_stocks
        self.h_dim = h_dim

        # Adaptive adjacency matrices E1^k and E2^k: (D x h_dim)
        self.E1 = nn.Parameter(torch.randn(in_channel, h_dim))  # D = in_channel (features)
        self.E2 = nn.Parameter(torch.randn(in_channel, h_dim))

        # GCN weight matrix: projects time steps (in_spatial) to h_dim
        self.gcn_weight = nn.Parameter(torch.randn(in_spatial, h_dim))

        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        # Input shape: [batch_size, num_stocks, features (D), time_steps (t)]
        b, n, c, t = x.size()

        # Compute local adjacency matrix R^k: (D x D)
        Rk = self.relu(torch.matmul(self.E1, self.E2.t()))
        # print('Pk',Rk.shape)
        Rk = self.softmax(Rk)

        # Add identity matrix and normalize
        I = torch.eye(c).to(x.device)
        Rk_tilde = Rk + I
        row_sum = torch.sum(Rk_tilde, dim=1)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(row_sum))
        normalized_adj = torch.matmul(torch.matmul(D_inv_sqrt, Rk_tilde), D_inv_sqrt)

        # Process each stock's features
        stock_embeddings = []
        for i in range(n):
            # Extract stock i's data: [batch, features, time]
            stock_data = x[:, i, :, :]  # Shape: [b, c, t]

            # Apply GCN propagation
            propagated = torch.matmul(normalized_adj, stock_data)  # [b, c, t]

            # Transform time dimension features
            transformed = torch.matmul(self.gcn_weight.t(), propagated )  # [b, c, h_dim]
            activated = self.relu(transformed)

            # Flatten to create stock embedding: [b, c*h_dim]
            stock_embeddings.append(activated.view(b, -1))

        # Compute cosine similarity between stock embeddings
        embeddings = torch.stack(stock_embeddings, dim=1)  # [b, n, c*h_dim]
        norms = torch.norm(embeddings, dim=2, keepdim=True)
        normalized = embeddings / (norms + 1e-8)

        # Batch-wise cosine similarity matrix
        global_relation = torch.bmm(normalized, normalized.transpose(1, 2))  # [b, n, n]
        # print('global',global_relation.shape)

        return global_relation
#######




class LSTMgate(nn.Module):
    def __init__(self, input_size, output_size, activation, stocks, ):
        super(LSTMgate, self).__init__()
        self.activation = activation
        self.W = Parameter(torch.FloatTensor(stocks, input_size, output_size))
        self.bias = Parameter(torch.zeros(stocks, output_size))
        self.reset_param(self.W)

    def reset_param(self, x):
        stdv = 1. / math.sqrt(x.size(1))
        x.data.uniform_(-stdv, stdv)

    def forward(self, x):
        print(f"x shape: {x.shape}")
        print(f"W shape: {self.W.shape}")
        x = torch.unsqueeze(x, -2)
        result = self.activation(torch.matmul(x, self.W).squeeze() + self.bias)
        print(f"Result shape after matmul: {result.shape}")
        return result

        # return self.activation(torch.matmul(x, self.W).squeeze() + self.bias)


class LSTMcell(nn.Module):
    def __init__(self, in_size, out_size, stocks):
        super(LSTMcell, self).__init__()
        self.in_size = in_size
        self.out_feat = out_size
        self.input = LSTMgate(in_size + out_size, out_size, nn.Sigmoid(), stocks)
        self.output = LSTMgate(in_size + out_size, out_size, nn.Sigmoid(), stocks)
        self.forget = LSTMgate(in_size + out_size, out_size, nn.Sigmoid(), stocks)
        self.candidate = LSTMgate(in_size + out_size, out_size, nn.Tanh(), stocks)
        # in_size = 32 - out_size  # 所以 in_size 应该是 16
        # self.input = LSTMgate(in_size + out_size, out_size, nn.Sigmoid(), stocks)
        # self.output = LSTMgate(in_size + out_size, out_size, nn.Sigmoid(), stocks)
        # self.forget = LSTMgate(in_size + out_size, out_size, nn.Sigmoid(), stocks)
        # self.candidate = LSTMgate(in_size + out_size, out_size, nn.Tanh(), stocks)

    def forward(self, xt, hidden, ct_1):  # hidden:t-1
        print(f"xt shape: {xt.shape}")
        print(f"hidden shape: {hidden.shape}")
        _, N, D = hidden.shape
        print(f"N: {N}")
        print(f"D: {D}")
        it = self.input(torch.cat([xt, hidden.expand(len(xt), N, D)], dim=-1))
        ot = self.output(torch.cat([xt, hidden.expand(len(xt), N, D)], dim=-1))
        ft = self.forget(torch.cat([xt, hidden.expand(len(xt), N, D)], dim=-1))
        chat = self.candidate(torch.cat([xt, hidden.expand(len(xt), N, D)], dim=-1))

        ct = ft * ct_1.expand(len(xt), N, D) + it * chat
        ht = ot * torch.tanh(ct)
        return ht, ct


class LSTM(nn.Module):
    def __init__(self, in_feat, out_feat, stocks):
        super(LSTM, self).__init__()
        self.in_feat = in_feat
        self.hid_size = out_feat
        self.stocks = stocks
        self.lstmcell = LSTMcell(in_feat, out_feat, stocks)

    #              B*T*N*D
    def forward(self, x, hidden=None, c=None):
        h = []
        if hidden == None:
            hidden = torch.zeros((1, self.stocks, self.hid_size), device=x.device, dtype=x.dtype)
            c = torch.zeros((1, self.stocks, self.hid_size), device=x.device, dtype=x.dtype)
        for t in range(len(x[0])):
            print(f"x[:, t] shape: {x[:, t].shape}")
            print(f"hidden shape: {hidden.shape}")
            hidden, c = self.lstmcell(x[:, t], hidden, c)
            h.append(hidden)
        return hidden.squeeze()




class Configs:
    def __init__(self, in_size, hidden_feat, dropout, embed, output_attention, factor, e_layers, n_heads, d_ff,
                 activation):
        self.enc_in = in_size
        # self.dec_in = in_size
        self.d_model = hidden_feat

        self.dropout = dropout
        self.embed = embed
        # self.pred_len = pred_len
        self.output_attention = output_attention
        self.factor = factor
        self.e_layers = e_layers
        # self.d_layers = d_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.activation = activation
        # self.c_out = c_out

    def copy(self):
        return Configs(
            in_size=self.enc_in,
            hidden_feat=self.d_model,
            dropout=self.dropout,
            embed=self.embed,
            output_attention=self.output_attention,
            factor=self.factor,
            e_layers=self.e_layers,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            activation=self.activation
        )



class SpatioTemporalFusion(nn.Module):
    def __init__(self, num_scales, feature_dim):
        super(SpatioTemporalFusion, self).__init__()
        self.num_scales = num_scales
        self.feature_dim = feature_dim

        # Linear transformation matrices for each scale
        self.scale_transforms = nn.ParameterList([
            nn.Parameter(torch.Tensor(feature_dim, feature_dim))
            for _ in range(num_scales)
        ])

        # Shared attention (gating) weight matrix
        self.attention_weights = nn.Parameter(torch.Tensor(2 * feature_dim, 1))

        # Layer Normalization for each scale except the coarsest
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(feature_dim)
            for _ in range(num_scales - 1)
        ])

        self._reset_parameters()

    def _reset_parameters(self):
        # Use Xavier uniform distribution to initialize parameters
        for w in self.scale_transforms:
            nn.init.xavier_uniform_(w)
        nn.init.xavier_uniform_(self.attention_weights)

    def forward(self, scale_features):
        """
        Parameters:
            scale_features: A list of tensors [P^1, P^2, ..., P^K] from fine to coarse scale.

        Returns:
            Fused representation at the finest scale.
        """
        # Start from the coarsest scale and iterate to the finest
        for k in range(self.num_scales - 1, -1, -1):
            # Perform linear transformation for the current scale
            P_k = scale_features[k] @ self.scale_transforms[k]

            if k < self.num_scales - 1:
                # Concatenate the current scale's original and transformed features with the next coarser scale
                combined = torch.cat([scale_features[k], P_prev], dim=-1)

                # Compute gating values using a sigmoid function
                alpha_k = torch.sigmoid(combined @ self.attention_weights)

                # Dynamic feature fusion using gating mechanism
                P_k = self.layer_norms[k](
                    alpha_k * scale_features[k] + (1 - alpha_k) * P_prev
                )

            # Store current transformed scale feature for the next iteration
            P_prev = P_k

        # Return the final fused representation at the finest scale
        return P_prev



# class SpatioTemporalFusion(nn.Module):
#     def __init__(self, num_scales, feature_dim, dropout=0.3):
#         super(SpatioTemporalFusion, self).__init__()
#         self.num_scales = num_scales
#         self.transforms = nn.ModuleList([
#             nn.Linear(feature_dim, feature_dim) for _ in range(num_scales)
#         ])
#         self.att_gate = nn.Linear(feature_dim * 2, feature_dim)
#         self.dropout = nn.Dropout(dropout)
#         self.layer_norms = nn.ModuleList([
#             nn.LayerNorm(feature_dim) for _ in range(num_scales - 1)
#         ])
#
#     def forward(self, feats):
#         # feats: list [P1..PK], P1 finest
#         P_prev = None
#         for k in reversed(range(self.num_scales)):
#             Pk = self.transforms[k](feats[k])
#             if P_prev is not None:
#                 # gating fusion
#                 cat = torch.cat([feats[k], P_prev], dim=-1)
#                 alpha = torch.sigmoid(self.att_gate(cat))
#                 Pk = self.layer_norms[k](alpha * feats[k] + (1 - alpha) * P_prev)
#                 Pk = self.dropout(Pk)
#             P_prev = Pk
#         return P_prev

class LSTMModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=configs.enc_in,
            hidden_size=configs.d_model,
            num_layers=configs.e_layers,
            dropout=configs.dropout if configs.e_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, x):
        output, _ = self.lstm(x)  # 输出形状: (batch_size, seq_len, hidden_size)
        return self.dropout(output)

class Model(nn.Module):
    def __init__(self, args, stocks, device='cuda'):
        super(Model, self).__init__()
        self.device = device
        self.tran_A = []
        self.down_sampling_layers = args.down_sampling_layers
        self.num_scales = args.down_sampling_layers + 1
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2).to(device)
        self.trend_pool = nn.AvgPool1d(args.kernel_size, 1, (args.kernel_size - 1) // 2).to(device)
        self.configs = Configs(args.in_size, args.hidden_feat, args.dropout, args.embed,
                               args.output_attention, args.factor, args.e_layers, args.nhead,
                               args.d_ff, args.activation)
        # self.configs=Configs()
        # print("Dropout value:", configs.dropout)

        # self.trend_encoder = nn.ModuleList([LSTM(16,args.hidden_feat,stocks) for l in range(self.down_sampling_layers + 1)])
        self.trend_encoder = nn.ModuleList(
            [TransformerModel(self.configs) for l in range(self.down_sampling_layers + 1)]).to(device)
        # self.configs = Configs(args.in_size, args.hidden_feat, args.dropout, args.embed,
        #                        args.output_attention, args.factor, args.e_layers, args.nhead,
        #                        args.d_ff, args.activation)

        # # 用LSTM替换Transformer
        # self.trend_encoder = nn.ModuleList(
        #     [LSTMModel(self.configs) for _ in range(self.down_sampling_layers + 1)]
        # ).to(device)
        #
        # # 用于动态创建的LSTM（如果需要不同输入维度）
        # self.dynamic_lstms = nn.ModuleList()

        self.global_r=RGA_Module(in_channel=args.in_size,in_spatial=args.in_size).to(device)
        # self.tgnns = nn.ModuleList()
        self.tgnns = nn.ModuleList()
        for l in range(args.down_sampling_layers + 1):
            layer = GraphConvolutionLayer(args.hidden_feat, args.hidden_feat, args.dropout, args.device)
            self.tgnns.append(layer)
        self.fusion = SpatioTemporalFusion(
            num_scales=self.num_scales,
            feature_dim=args.hidden_feat  # feature_dim应为hidden_feat的维度
        )
        # self.fusion=SpatioTemporalFusion(num_scales=down_sampling_layers, feature_dim=feature_dim)
        # for l in range(self.down_sampling_layers + 1):
        #     # 创建一个 GraphConvolutionLayer 实例，可以在这里基于 l 来定制初始化参数
        #     layer = GraphConvolutionLayer(args.hidden_feat, args.hidden_feat, args.dropout, args.device)
        #
        #     # 添加到 ModuleList 中
        #     self.tgnns.append(layer)

        # self.sgnns = nn.ModuleList(
        #     [GraphConvolutionLayer(args.hidden_feat, args.hidden_feat) for l in range(self.down_sampling_layers + 1)])

        self.attfusion = args.attfusion

        if args.task == 'price' or args.task == 'ranking':
            self.predict = nn.Linear(args.hidden_feat * (args.down_sampling_layers + 1), 2).to(device)
        else:
            # self.predict = nn.Linear(args.hidden_feat * (args.down_sampling_layers + 1), 2).to(device)
            self.predict = nn.Linear(args.hidden_feat, 2).to(device)




    def forward(self, x):
        x = x.to(self.device)
        embedding_list = []
        trend_list = []
        # seasonal_list = []
        t = []
        # s = []
        #  print('x.shape:',x.shape)

        embedding_list.append(x)
        adjmatrix_list=[]
        B, N, D, _ = x.shape
        # print('x:',x.shape)
#######多尺度
        for i in range(self.down_sampling_layers+1):
            # 从原始 embedding_list[0] 中取数据
            # print(i)
            x_transpose = embedding_list[0].reshape((B * N * D), -1).to(self.device)  # (BND)T

            # 每次循环缩短原来的一半
            pool_layer = nn.AvgPool1d(kernel_size=2 ** (i + 1), stride=2 ** (i + 1))
            x_transpose = pool_layer(x_transpose)  # 每次减去一半的长度
            x_transpose = x_transpose.reshape(B, N, D, -1)
            # print('x_transpose.shape',x_transpose.shape)
            # print('x_transpose.shape',x_transpose.shape)

            embedding_list.append(x_transpose)
            # self.plot_changes_across_resolutions(embedding_list, feature_index=0)

            out=self.global_r(x_transpose)
            # out = self.adjv(x_transpose)  # 不再传递邻接矩阵
            # print('outadj.shape:',out.shape)
            adjmatrix_list.append(out)



        trend_trans_list = []

        for i, embedding in enumerate(embedding_list):
            # print('1',self.trend_encoder[i](embedding.transpose(1, 2)).shape)
            # print('embedding.',embedding.shape)
            B, N, _, D = embedding.shape  # 取出当前形状的值
            # print('111:',B,N,_,D)#16 196 5 16
            current_configs = Configs(
                self.configs.enc_in,  # You may want to modify this if needed
                self.configs.d_model,
                self.configs.dropout,
                self.configs.embed,
                self.configs.output_attention,
                self.configs.factor,
                self.configs.e_layers,
                self.configs.n_heads,
                self.configs.d_ff,
                self.configs.activation
            )
            current_configs.enc_in = D * _
            # 创建新的 TransformerModel 实例
            transformer_model = TransformerModel(current_configs)
            # print(transformer_model(embedding.transpose(1, 2).reshape(B, N, -1)).shape)
            trend_trans_list.append(transformer_model(embedding.transpose(1, 2).reshape(B, N, -1)))

        # trend_trans_list = []
        # for i, embedding in enumerate(embedding_list):
        #     B, N, _, D = embedding.shape
        #     # 调整输入维度配置
        #     current_configs = self.configs.copy()
        #     current_configs.enc_in = D * _
        #
        #     # 创建新的LSTM并注册到网络
        #     lstm_model = LSTMModel(current_configs).to(embedding.device)
        #     self.dynamic_lstms.append(lstm_model)  # 确保参数可学习
        #
        #     # 调整输入形状
        #     reshaped_input = embedding.transpose(1, 2).reshape(B, N, -1)
        #
        #     # 通过LSTM处理
        #     lstm_output = lstm_model(reshaped_input)
        #     trend_trans_list.append(lstm_output)

        for i, (x, A) in enumerate(zip(trend_trans_list, adjmatrix_list)):
            # print('111',x.shape)
            # in_features = x.shape[2]
            # x_reshaped = x.view(256, 196, -1)
            # print('A',A.shape,x.shape)
            # print(i)
            # print('t:', self.tgnns[i](x, A).shape)
            t.append(self.tgnns[i](x, A))
        # for i, tensor in enumerate(t):
        #     print(f"Shape of tensor {i}: {tensor.shape}")


        #####
        fused_features = self.fusion(t[::-1])

# #####


        # t = torch.cat(t, -1)
        # print('t:',t.shape)
        # t = t.view(t.shape[0], -1)  # 将t展平为2D张量
        out = self.predict(fused_features)
        # out = self.predict(t)

        return out
