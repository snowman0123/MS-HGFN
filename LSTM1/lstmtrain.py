import time
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, mean_squared_error, mean_absolute_error
from torch import nn, optim
import os
from DataProcess import *
from utils import *
import matplotlib.pyplot as plt
from LSTM import Model
torch.set_printoptions(precision=5, sci_mode=False)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from torch import nn, optim

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda:0',
                    help='GPU To Use')

parser.add_argument('--market', type=str, default='CSI300',
                    help='Market Index Information')

parser.add_argument('--epochs', type=int, default=100,
                    help='Training max epoch')

parser.add_argument('--lr', type=float, default=0.05,
                    help='Learning rate')

parser.add_argument('--weight_constraint', type=float, default=5e-4,
                    help='L2 Weight Constraint')

parser.add_argument('--time_length', type=int, default=10,
                    help='time length')

parser.add_argument('--hidden_feat', type=int, default=64, )

parser.add_argument('--task', type=str, default='price', )

parser.add_argument('--dropout', type=float, default=.5, )

parser.add_argument('--batch_size', type=int, default=32)

parser.add_argument('--datanormtype', type=str, default='meanstd')

parser.add_argument('--attention', type=str, default=False,
                    help='GPU To Use')
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def rank(predict, label, k):
    pindices = torch.argsort(predict, descending=True)
    tindices = torch.argsort(label, descending=True)
    return (torch.isin(pindices[:k], tindices[:k]).sum().float() / k).item()


def test(model, dataloader):
    predicts = None
    labels = None
    model.eval()
    for data, label in dataloader:
        model.eval()
        out = model(data).cpu().detach()
        label = label.cpu().detach()
        if predicts == None:
            predicts = out
            labels = label
        else:
            predicts = torch.cat([predicts, out], dim=0)
            labels = torch.cat([labels, label], dim=0)
    print(F.mse_loss(predicts.reshape(-1),labels.reshape(-1)))



def train(args, model, train_loader):
    for epoch in range(args.epochs):
        avg_loss = 0
        model.train()
        print('epoch ' + str(epoch), end=':')
        for data, label in train_loader:
            optimizer.zero_grad()
            if len(data) < batch_size:
                break
            out = model(data)
            loss = criterion(out, label)
            avg_loss += loss
            loss.backward()
            optimizer.step()

        avg_loss = avg_loss / len(train_loader)
        print(avg_loss)



if __name__ == '__main__':
    dic = {}
    set_random_seed(1)

    args = parser.parse_args()
    DEVICE = args.device
    batch_size = args.batch_size
    time_length = args.time_length
    hidden_feat = args.hidden_feat
    train_loader, val_loader, test_loader = LoadData('SP100.npy', batch_size,
                                                               time_length,
                                                               args.datanormtype, args.task, args.device)

    _, stocks, in_feat = get_dimension(train_loader)
    criterion = nn.MSELoss()
    model = Model(args, in_feat, hidden_feat, time_length, stocks)
    model.cuda(device=DEVICE)
    model = model.float()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_constraint)
    train(args, model, train_loader)
    test(model,test_loader)

