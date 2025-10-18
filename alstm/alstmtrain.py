import time
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, mean_squared_error, mean_absolute_error
from torch import nn, optim
import os
from DataProcess import *
from alstm.model import Model
from evaluate import daily_evaluate1, evaluate

from utils import *


torch.set_printoptions(precision=5, sci_mode=False)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from torch import nn, optim

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda:0',
                    help='GPU To Use')

parser.add_argument('--epochs', type=int, default=100,
                    help='Training max epoch')

parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate')

parser.add_argument('--weight_constraint', type=float, default=5e-4,
                    help='L2 Weight Constraint')

parser.add_argument('--time_length', type=int, default=10,
                    help='time length')

parser.add_argument('--gcnhidden_feat', type=int, default=32,
                    help='gcnhidden_feat')

parser.add_argument('--hidden_feat', type=int, default=32, )

parser.add_argument('--task', type=str, default='trend', )

parser.add_argument('--alpha', type=float, default=1., )

parser.add_argument('--dropout', type=float, default=.5, )

parser.add_argument('--batch_size', type=int, default=64)

parser.add_argument('--datanormtype', type=str, default='meanstd')

parser.add_argument('--in_size', type=int, default=5)

parser.add_argument('--T', type=int, default=1)

parser.add_argument('--expect', type=float, default=0.00)

parser.add_argument('--gnnmodel', type=str, default='gcn')


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def test(model, dataloader):
    predicts = None
    labels = None
    model.eval()
    for data, label in dataloader:
        model.eval()
        out = model(data['indicator'])
        out = torch.argmax(out.cpu().detach(), dim=-1).reshape(-1)
        label = label.cpu().detach().reshape(-1)
        if predicts == None:
            predicts = out
            labels = label
        else:
            predicts = torch.cat([predicts, out])
            labels = torch.cat([labels, label])
    print("比例是:{:.2f}".format((torch.sum(torch.eq(labels, 1)) / len(labels)).item() * 100), '%', end=' ')
    print("精度是:{:.2f}".format(accuracy_score(predicts, labels) * 100), '% ',
          'MCC是:{:.4f}'.format(matthews_corrcoef(predicts, labels)))


def train(args, model, train_loader):
    for epoch in range(args.epochs):
        avg_loss = 0
        model.train()
        # print('epoch ' + str(epoch), end=':')
        for data, label in train_loader:
            optimizer.zero_grad()
            # print(data['indicator'].shape,data['future'].shape,label.shape)
            out = model(data['indicator'])

            _, _, c = out.shape
            out = out.reshape(-1, c)
            label = label.reshape(-1)
            loss = criterion(out, label)

            avg_loss += loss
            loss.backward()
            optimizer.step()

        avg_loss = avg_loss / len(train_loader)
        # print(avg_loss)


if __name__ == '__main__':
    # set_random_seed(1)
    for i in range(10):
        args = parser.parse_args()
        DEVICE = args.device
        batch_size = args.batch_size
        time_length = args.time_length
        hidden_feat = args.hidden_feat
        A = torch.from_numpy(np.load('sp100_graph_relation.npy')).float().to(DEVICE)
        n1, n2 = A.shape
        A = A - torch.eye(n1).to(A.device)
        train_loader, val_loader, test_loader = LoadData('SP100.npy', batch_size,
                                                         time_length,
                                                         args.datanormtype, args.task, args.T, args.expect, args.device)

        _, stocks, in_feat = get_dimension(train_loader)
        if args.task == 'future':
            criterion = nn.CrossEntropyLoss()
        if args.task == 'price':
            criterion = nn.MSELoss()
        if args.task == 'trend':
            criterion = nn.CrossEntropyLoss()

        model = Model(args.in_size, hidden_feat, time_length, stocks, args.task)
        model.cuda(device=DEVICE)
        model = model.float()

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_constraint)

        train(args, model, train_loader)

        test(model, train_loader)
        # test(smodel, val_loader)
        test(model, test_loader)
        daily_evaluate1(model,test_loader)