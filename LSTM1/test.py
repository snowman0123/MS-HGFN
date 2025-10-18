import time

import networkx as nx
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, mean_squared_error, mean_absolute_error
from torch import nn, optim
import os
from DataProcess import *
from utils import *
from HDHGNN import *
import matplotlib.pyplot as plt
from similarity_index_of_label_graph_package import similarity_index_of_label_graph_class
from networkx.generators.directed import gnr_graph
torch.set_printoptions(precision=5, sci_mode=False)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from torch import nn, optim

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda:0',
                    help='GPU To Use')

parser.add_argument('--market', type=str, default='CSI300',
                    help='Market Index Information')

parser.add_argument('--model', type=str, default='hgnnmyloss', )

parser.add_argument('--epochs', type=int, default=5,
                    help='Training max epoch')

parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning rate')

parser.add_argument('--task', type=str, default='ranking',
                    help='ranking,priceprediction')

parser.add_argument('--loss', type=str, default='myloss',
                    help='myloss,oldrankloss,mse,crossentropy')

parser.add_argument('--weight_constraint', type=float, default=5e-4,
                    help='L2 Weight Constraint')

parser.add_argument('--time_length', type=int, default=10,
                    help='time length')

parser.add_argument('--in_market', type=int, default=5,
                    help='in_market')

parser.add_argument('--hidden_feat', type=int, default=64, )

parser.add_argument('--top_stocks', type=int, default=5, )

parser.add_argument('--dropout', type=float, default=.5, )

parser.add_argument('--lstmattention', type=bool, default=True)

parser.add_argument('--batch_size', type=int, default=32)

parser.add_argument('--datanormtype', type=str, default='meanstd')


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def rank(predict, label, k):
    pindices = torch.argsort(predict, descending=True)
    tindices = torch.argsort(label, descending=True)
    return (torch.isin(pindices[:k], tindices[:k]).sum().float() / k).item()
def laplacian_spectrum_from_adjacency(adj_matrix):
    # 从邻接矩阵创建图
    graph = nx.from_numpy_array(adj_matrix)
    # 计算拉普拉斯矩阵
    L = nx.laplacian_matrix(graph).toarray()
    # 计算特征值
    eigenvalues = np.linalg.eigvalsh(L)
    return sorted(eigenvalues)

def compare_graphs(adj_matrix1, adj_matrix2):
    # 获取两个图的拉普拉斯谱
    spec1 = laplacian_spectrum_from_adjacency(adj_matrix1)
    spec2 = laplacian_spectrum_from_adjacency(adj_matrix2)

    # 比较特征值
    distance = np.linalg.norm(np.array(spec1) - np.array(spec2))
    return distance

def plott(H):
    rows, cols = H.shape
    plt.figure(figsize=(5, 10))

    colors = ['#ff007f', '#ffff00', '#007fff', '#00ff7f', '#7f00ff']  # Define colors for each column
    colors = ['#E9686F', '#F9D95C', '#66EDF6', '#5CCBCE', '#8663AA']

    # Plot points where matrix element is 1
    k = 0
    for i in range(rows):
        for j in range(cols):
            if H[i, j] == 1:
                plt.scatter(j, i, color=colors[j % len(colors)], s=100)
                k += 1

    plt.xlabel('Market Influences', fontsize=18)
    plt.ylabel('Stocks', fontsize=22)
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    plt.gca().invert_yaxis()  # Invert y-axis to match matrix coordinates
    # plt.show()
    plt.savefig('m.pdf', format='pdf')


def test(model, dataloader, i=True):
    H=np.load('sse100hypergraph.npy')
    model.eval()
    for data, label in dataloader:
        model.eval()
        pearsonInfluence = model.getinfluence(data['stock'], data['market'])
        x = model.stocktemporal(data['stock'], data['market'])
        h, (m, c) = model.markefeat(data['market'])
        out = model.marketrelational.get_incidenceHW(x,pearsonInfluence)[0].cpu().detach()
        plott(out[0])


        # print(out[0][0].shape)
        # for i in range(3):
        #     plt.imshow(out[i], cmap='Blues', interpolation='nearest')
        #     plt.savefig(str(i)+'matrix_plot.pdf', format='pdf')

        #[14, 16 ,26 ,38 ,41,43, 49, 51 ,55 ,59, 61]
        for i in range(14):
            print(i,end=':')
            for j in range(70):
                if H[j,i]==1:
                    print(j, end=' ')
            print('')
        print('--------')
        for i in [0,3,7,11,15,19]:
            for j in range(70):
                if out[0,j,i]==1:
                    print(j,end=' ')
            print('')



        break




def evaluate(model, dataloader, top_stocks):
    predicts = labels = None
    model.eval()
    for data, label in dataloader:
        model.eval()
        out = model(data['stock'], data['market']).cpu().detach()
        label = label.cpu().detach()
        if predicts == None:
            predicts = out
            labels = label
        else:
            predicts = torch.cat([predicts, out], dim=0)
            labels = torch.cat([labels, label], dim=0)
    # print(predicts.shape, labels.shape)  # T*N
    # sharp=[]
    roi = []
    rand = []
    sharp = []
    rankk = []

    # rank_score=[]
    for i in range(len(predicts)):
        pred_sort = torch.argsort(predicts[i], descending=True)
        # labels_sort = torch.argsort(labels[i], descending=True)

        pred_topN = pred_sort[:top_stocks]
        topN = labels[i, pred_topN]
        roi.append(torch.mean(topN).item())

        rand_topN = list(range(len(predicts[i])))
        random.shuffle(rand_topN)
        randtopN = labels[i, rand_topN[:top_stocks]]
        rand.append(torch.mean(randtopN).item())

        sharp.append(
            torch.mean(topN) / torch.std(topN) if torch.abs(torch.std(topN)) > 1e-3 else torch.mean(topN) / 1e-3)
        rankk.append(rank(predicts[i], labels[i], top_stocks))

        # rank10.append(rank(predicts[i], labels[i], 10))
        # rank20.append(rank(predicts[i], labels[i], 20))
        # rank30.append(rank(predicts[i], labels[i], 30))

    res = 'MSE:{:.6f},ROI:{:.6f},SP:{:.4f},rank5:{:.4f}\n'.format(
        mean_squared_error(predicts.reshape(-1).detach().numpy(), labels.reshape(-1).detach().numpy()),
        sum(roi), sum(sharp) / len(sharp), sum(rankk) / len(rankk)
    )
    print(res,end='')
    return res




if __name__ == '__main__':
    dic = {}
    for topk in [1]:
        mse = []
        roi = []
        sharp = []
        rankrate = []
        # for i in range(5):
        args = parser.parse_args()
        DEVICE = args.device
        batch_size = args.batch_size
        time_length = args.time_length
        hidden_feat = args.hidden_feat
        modelname = args.model
        train_loader, val_loader, test_loader = LoadData('SSE100.npy', 'SSE100_market.npy', batch_size,
                                                                   time_length,
                                                                   args.datanormtype, args.task, args.device)

        _, stocks, in_feat = get_dimension(train_loader)

        if args.loss == 'mse':
            criterion = nn.MSELoss()
        elif args.loss == 'myloss':
            criterion = MyLoss()
        elif args.loss == 'oldrankloss':
            criterion = RankingLoss()
        elif args.loss == 'crossentropy':
            criterion = nn.CrossEntropyLoss()

        model = Model(args, in_feat, hidden_feat, time_length, stocks)
        model.load_state_dict(torch.load('top1.pth'))
        model.cuda(device=DEVICE)
        model = model.float()
        test(model,test_loader)