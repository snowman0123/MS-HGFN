import copy
import time
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, mean_squared_error, mean_absolute_error
from torch import nn, optim
import os
from DataProcess import *
from Teacher_Student.model import Teacher, Student
from evaluate import evaluate
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

parser.add_argument('--epochs', type=int, default=100,
                    help='Training max epoch')

parser.add_argument('--lr', type=float, default=0.0005,
                    help='Learning rate')

parser.add_argument('--weight_constraint', type=float, default=5e-4,
                    help='L2 Weight Constraint')

parser.add_argument('--time_length', type=int, default=20,
                    help='time length')

parser.add_argument('--hidden_feat', type=int, default=32, )

parser.add_argument('--task', type=str, default='future', )

parser.add_argument('--alpha', type=float, default=1., )

parser.add_argument('--dropout', type=float, default=.5, )

parser.add_argument('--batch_size', type=int, default=64)

parser.add_argument('--datanormtype', type=str, default='meanstd')

parser.add_argument('--in_size', type=int, default=5)

parser.add_argument('--T', type=int, default=20, help='20 10 5')

parser.add_argument('--expect', type=float, default=0.04, help='0.04, 0.025, 0.015')

parser.add_argument('--gnnmodel', type=str, default='GAT')

parser.add_argument('--gnnteachermodel', type=str, default='GCN')


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def test2(model, dataloader, A):
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
          'MCC是:{:.4f}'.format(matthews_corrcoef(predicts, labels)))


def test(model, tmodel, dataloader, A):
    predicts = None
    labels = None
    model.eval()
    tmodel.eval()

    for data, label in dataloader:

        out = tmodel.pred(
            model.lstm(data['indicator']) + args.alpha * model.gnn_model(model.lstm(data['indicator']), A))

        # out, _ = model(data['indicator'], A)
        out = torch.argmax(out.cpu().detach(), dim=-1).reshape(-1)
        label = label.cpu().detach().reshape(-1)
        if predicts == None:
            predicts = out
            labels = label
        else:
            predicts = torch.cat([predicts, out])
            labels = torch.cat([labels, label])
    # print("比例是:{:.2f}".format((torch.sum(torch.eq(labels, 1)) / len(labels)).item() * 100), '%', end=' ')
    print("精度是:{:.2f}".format(accuracy_score(predicts, labels) * 100), '% ',
          'MCC是:{:.4f}'.format(matthews_corrcoef(predicts, labels)))
    return accuracy_score(predicts, labels)


def train(args, smodel, tmodel, train_loader,model):
    bestacc = 0

    for epoch in range(args.epochs):
        avg_loss = 0
        print('epoch ' + str(epoch), end='\n')
        for data, label in train_loader:
            smodel.train()
            optimizer.zero_grad()
            # print(data['indicator'].shape,data['future'].shape,label.shape)
            sout, sembedding = smodel(data['indicator'], A, tmodel)
            tout, _, tembedding = tmodel(data['indicator'], data['future'].transpose(-2, -1), A)

            diss_loss = cri(sembedding.reshape(-1,sembedding.shape[-1]), tembedding.reshape(-1,sembedding.shape[-1]))

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
        print('val:', end='')
        acc = test(smodel, tmodel, val_loader, A)
        print('test:', end='')
        test(smodel, tmodel, test_loader, A)

        if acc > bestacc:
            bestacc = acc
            model.load_state_dict(smodel.state_dict())
        test(model, tmodel, test_loader, A)
        print('\n', avg_loss)
def mean_and_variance(numbers):
    mean = sum(numbers) / len(numbers)
    variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
    return mean, variance

if __name__ == '__main__':
    roi = []
    sharp = []
    mdd = []
    invest = None
    for i in range(5):
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
            # criterion=HSIC()
        if args.task == 'price':
            criterion = nn.MSELoss()

        cri=HSIC()

        smodel = Student(args.in_size, hidden_feat, time_length, stocks, args.gnnmodel, args.task)
        smodel.cuda(device=DEVICE)
        smodel = smodel.float()

        tmodel = Teacher(args.in_size, hidden_feat, time_length, stocks, args.gnnteachermodel, args.task, args.T)
        tmodel.load_state_dict(torch.load('teacher_model.pth'))
        tmodel.cuda(device=DEVICE)
        tmodel.eval()

        model=Student(args.in_size, hidden_feat, time_length, stocks, args.gnnmodel, args.task)
        model.cuda(device=DEVICE)
        model = model.float()

        optimizer = optim.Adam(smodel.parameters(), lr=args.lr, weight_decay=args.weight_constraint)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                               patience=30,
                                                               factor=0.1,
                                                               verbose=True)

        train(args, smodel, tmodel, train_loader,model)

        test(model, tmodel, train_loader, A)
        test(model, tmodel, val_loader, A)
        test(model, tmodel, test_loader, A)

        a, b, c, investment_price = evaluate(model, tmodel, test_loader, args.T, args.expect, A)
        roi.append(a)
        sharp.append(b)
        mdd.append(c)
        investment_price = np.array(investment_price)
        if i == 0:
            invest = investment_price
        else:
            invest += investment_price

    print(mean_and_variance(roi))
    print(mean_and_variance(sharp))
    print(mean_and_variance(mdd))
    invest = invest / 5
    print(invest.tolist())
    plt.plot(invest, linestyle='-', color='b', label='Data')
    plt.legend()
    plt.grid(True)
    plt.show()
