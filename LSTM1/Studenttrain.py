import time
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, mean_squared_error, mean_absolute_error
from torch import nn, optim
import os
from DataProcess import *
from Teacher_Student.model import Teacher, Student
from utils import *
import matplotlib.pyplot as plt
from LSTM import *

torch.set_printoptions(precision=5, sci_mode=False)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from torch import nn, optim

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda:0',
                    help='GPU To Use')

parser.add_argument('--market', type=str, default='CSI300',
                    help='Market Index Information')

parser.add_argument('--epochs', type=int, default=500,
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

parser.add_argument('--task', type=str, default='future', )

parser.add_argument('--alpha', type=float, default=1., )

parser.add_argument('--dropout', type=float, default=.5, )

parser.add_argument('--batch_size', type=int, default=64)

parser.add_argument('--datanormtype', type=str, default='meanstd')

parser.add_argument('--in_size', type=int, default=5)


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
        out, _ = model(data['indicator'], A)
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
          'MCC是:{:.2f}'.format(matthews_corrcoef(predicts, labels)))


def train(args, smodel, tmodel, train_loader):
    for epoch in range(args.epochs):
        avg_loss = 0
        smodel.train()
        print('epoch ' + str(epoch), end=':')
        for data, label in train_loader:
            # print(data['indicator'].shape,label.shape)
            optimizer.zero_grad()
            # if len(data) < batch_size:
            #     break
            # print(data['indicator'].shape,data['future'].shape,label.shape)
            sout, sembedding = smodel(data['indicator'], A)
            tout, tembedding = tmodel(data['indicator'], data['future'].transpose(-2, -1), A)
            diss_loss = F.mse_loss(sembedding, tembedding)

            _, _, c = sout.shape
            sout = sout.reshape(-1, c)
            label = label.reshape(-1)

            pred_loss = criterion(sout, label)
            loss = pred_loss + args.alpha * diss_loss
            avg_loss += loss
            loss.backward()
            optimizer.step()

        avg_loss = avg_loss / len(train_loader)
        scheduler.step(avg_loss)
        print(avg_loss)


if __name__ == '__main__':
    # set_random_seed(1)
    args = parser.parse_args()
    DEVICE = args.device
    batch_size = args.batch_size
    time_length = args.time_length
    hidden_feat = args.hidden_feat
    A = torch.from_numpy(np.load('sp100_graph_relation.npy')).float().to(DEVICE)
    n1, n2 = A.shape
    A = A - torch.eye(n1).to(A.device)
    T = 20
    expect = 0.04
    gnn = 'GCN'
    train_loader, val_loader, test_loader = LoadData('SP100.npy', batch_size,
                                                     time_length,
                                                     args.datanormtype, args.task, T, expect, args.device)

    _, stocks, in_feat = get_dimension(train_loader)
    if args.task == 'future':
        criterion = nn.CrossEntropyLoss()
    if args.task == 'price':
        criterion = nn.MSELoss()

    smodel = Student(args.in_size, hidden_feat, time_length, stocks, gnn, args.task)
    smodel.cuda(device=DEVICE)
    smodel = smodel.float()

    tmodel = Teacher(args.in_size, hidden_feat, time_length, stocks, gnn, args.task, T)
    torch.load('teacher_model.pth')
    tmodel.cuda(device=DEVICE)
    tmodel.eval()

    optimizer = optim.Adam(smodel.parameters(), lr=args.lr, weight_decay=args.weight_constraint)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                           patience=30,
                                                           factor=0.1,
                                                           verbose=True)
    test(smodel, test_loader)

    train(args, smodel, tmodel, train_loader)

    test(smodel, train_loader)
    test(smodel, test_loader)
