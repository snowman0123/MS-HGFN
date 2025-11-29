import copy
import time
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, matthews_corrcoef, recall_score, f1_score, mean_squared_error, mean_absolute_error, roc_auc_score
from torch import nn, optim
import os
from sklearn.model_selection import KFold, StratifiedKFold
from collections import Counter
from tqdm import tqdm

from DataProcess import *
from Teacher_Student.model import Teacher, Student
from evaluate import evaluate
from model import Model, EarlyStopping
from utils import *
import matplotlib.pyplot as plt
from LSTM import *

torch.set_printoptions(precision=5, sci_mode=False)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from torch import nn, optim

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda',
                    help='Device to use for computation')

parser.add_argument('--epochs', type=int, default=20,
                    help='Training max epoch')

parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate')

parser.add_argument('--weight_constraint', type=float, default=0,
                    help='L2 Weight Constraint')

parser.add_argument('--time_length', type=int, default=16,
                    help='time length')

parser.add_argument('--hidden_feat', type=int, default=16)

parser.add_argument('--task', type=str, default='trend')

parser.add_argument('--dropout', type=float, default=.5)

parser.add_argument('--batch_size', type=int, default=1)

parser.add_argument('--datanormtype', type=str, default='meanstd')

parser.add_argument('--down_sampling_layers', type=int, default=2)

parser.add_argument('--num_layers', type=int, default=2)

parser.add_argument('--num', type=int, default=2)

parser.add_argument('--kernel_size', type=int, default=3)

parser.add_argument('--in_size', type=int, default=5)

parser.add_argument('--nhead', type=int, default=1)

parser.add_argument('--attfusion', type=bool, default=True)

parser.add_argument('--d_model', type=int, default=16)

# parser.add_argument('--d_ff', type=int, default=16)

# parser.add_argument('--dmodel', type=int, default=16)

parser.add_argument('--output_attention', type=bool, default=False)

parser.add_argument('--activation', type=str, default='relu')

parser.add_argument('--embed', type=str, default='fixed')

parser.add_argument('--freq', type=str, default='t', help='时间特征的频率')

parser.add_argument('--factor', type=int, default=5, help='注意力的缩放因子')

parser.add_argument('--hidden_size', type=int, default=5)

parser.add_argument('--max_len', type=int, default=16)

parser.add_argument('--d_ff', type=int, default=2, help='dimension of fcn')

parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')

parser.add_argument('--time_step', type=int, default=8)

parser.add_argument('--augmentation', type=bool, default=True)

parser.add_argument('--T', type=int, default=1)

parser.add_argument('--expect', type=float, default=0.04, help='0.04, 0.025, 0.015')





def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def runtest(dataloader):
    predicts = None
    labels = None
    all_probs = None
    # adj_matrices = []  # 初始化邻接矩阵列表
    early_stopping.best_model.eval()

    for data, label,actually_price in dataloader:
        data, label = data.to('cuda:0'), label.to('cuda:0')
        out = model(data.permute(0, 2, 3, 1).to('cuda:0'))

        out = out.squeeze(0)
        # 保存完整的概率分布以计算AUC
        probabilities = torch.softmax(out, dim=-1)
        class_probs = probabilities[:, 1]  # 基于阳性类提取概率，假设阳性类是1

        # 选择最大值的索引（选择类别）
        out = torch.argmax(out, dim=-1)  # 结果为 [196]

        # 将标签整平到一维
        label = label.reshape(-1)
        if predicts == None:
            predicts = out
            labels = label
            all_probs = class_probs
        else:
            predicts = torch.cat([predicts, out])
            labels = torch.cat([labels, label])
            all_probs = torch.cat([all_probs, class_probs])

    predicts = predicts.cpu().numpy() if predicts.is_cuda else predicts.numpy()
    labels = labels.cpu().numpy() if labels.is_cuda else labels.numpy()
    all_probs = all_probs.detach().cpu().numpy()  # 使用 detach() 以分离梯度计算图

    # 计算AUC值
    try:
        auc = roc_auc_score(labels, all_probs) * 100  # AUC 换算为百分比表示
    except ValueError as e:
        auc = None
        print(f"计算AUC时遇到错误: {e}")

    
    print(f"比例是: {positive_ratio:.2f}%", end=' ')

    # print("比例是:{:.2f}".format((torch.sum(torch.eq(labels, 1)) / len(labels)).item() * 100), '%', end=' ')
    print("精度是:{:.2f}".format(accuracy_score(predicts, labels) * 100), '% ',
          'MCC是:{:.4f}'.format(matthews_corrcoef(predicts, labels)))






def runtrain(args, train_loader, val_loader):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    early_stopping = EarlyStopping(args, device, stocks, patience=5)  # 确保已正确初始化

    for epoch in range(args.epochs):
        # === 训练阶段 ===
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for data, labels, _ in tqdm(train_loader):  # 规范命名
            data = data.permute(0, 2, 3, 1).to(device)
            labels = labels.squeeze().long().to(device)

            optimizer.zero_grad()
            outputs = model(data)

            # 维度校验保护
            if outputs.dim() > 2:
                outputs = outputs.view(-1, outputs.size(-1))
            labels = labels.view(-1)

            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # 累计指标
            train_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        # === 验证阶段 ===
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for data, labels, _ in val_loader:
                data = data.permute(0, 2, 3, 1).to(device)
                labels = labels.squeeze().long().to(device)

                outputs = model(data)
                if outputs.dim() > 2:
                    outputs = outputs.view(-1, outputs.size(-1))
                labels = labels.view(-1)

                val_loss += criterion(outputs, labels).item() * data.size(0)

        # === 指标计算 ===
        train_loss = train_loss / total
        val_loss = val_loss / len(val_loader.dataset)
        train_acc = correct / total

        # === 早停机制 ===
        early_stopping(val_loss, model,epoch)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}")
            # model.load_state_dict(torch.load('checkpoint.pt'))  # 恢复最佳参数

            break





if __name__ == '__main__':
    set_random_seed(3407)
    MSE = []
    MAE = []
    ROI_list = []
    Sharpe_list = []
    P_ak_list = []


    class FocalLoss(nn.Module):
        def __init__(self, alpha=1, gamma=2, reduction='mean'):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction

        def forward(self, inputs, targets):
            CE_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-CE_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * CE_loss

            return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()


    for i in range(1):
       
        args = parser.parse_args()
        DEVICE = args.device
        batch_size = args.batch_size
        time_length = args.time_length
        hidden_feat = args.hidden_feat
    
        train_loader, val_loader, test_loader = LoadData('/home/xxs/trend/CSI500.npy', batch_size,
                                                         time_length,
                                                         args.datanormtype, args.task, args.device)
        


        def calculate_class_weights(train_loader):
            # 统计训练集中每个类别的样本数
            label_counts = Counter()
            for _, labels, _ in train_loader:
                # 累积计数器中各类别样本数
                label_counts.update(labels.flatten().tolist())

            # 计算每个类别的权重
            total_samples = sum(label_counts.values())
            num_classes = len(label_counts)
            weights = np.zeros(num_classes)

            for label, count in label_counts.items():
                weights[label] = total_samples / (num_classes * count)

            # 转化为 tensor 传入损失函数
            return torch.tensor(weights, dtype=torch.float32).to(args.device)


        # 在定义损失函数之前，计算权重
        class_weights = calculate_class_weights(train_loader)
        # criterion = nn.CrossEntropyLoss(weight=class_weights)

        _, stocks, in_feat = get_dimension(train_loader)
        if args.task == 'trend':
            num_classes = 2  # 设置为类别数量
            # criterion = nn.CrossEntropyLoss(weight=class_weights)
            criterion = FocalLoss(alpha=class_weights[1], gamma=2)

        if args.task == 'price' or args.task == 'ranking':
            num_classes = 1
            criterion = nn.MSELoss()


        device = "cuda:0"


        model = Model(args, stocks, device)

        model.cuda()
        model = model.float()
        early_stopping = EarlyStopping(args, DEVICE, stocks)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_constraint)
       
        runtrain(args, train_loader, val_loader)
        
        runtest(test_loader)
        
       
        

    
