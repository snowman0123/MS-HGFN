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
import shap
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

parser.add_argument('--time_length', type=int, default=30,
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

parser.add_argument('--T', type=int, default=20,help='20 10 5')

parser.add_argument('--expect', type=float, default=0.04,help='0.04, 0.025, 0.015')

parser.add_argument('--gnnmodel', type=str, default='GAT')


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
          'MCC是:{:.4f}'.format(matthews_corrcoef(predicts, labels)))




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
    train_loader, val_loader, test_loader = LoadData('SP100.npy', batch_size,
                                                     time_length,
                                                     args.datanormtype, args.task, args.T, args.expect, args.device)


    _, stocks, in_feat = get_dimension(train_loader)


    smodel = Student(args.in_size, hidden_feat, time_length, stocks, args.gnnmodel, args.task)
    smodel.load_state_dict(torch.load('student_model.pth'))
    smodel.cuda(device=DEVICE)
    smodel.eval()

    for data,_ in train_loader:
        # explain predictions of the model on four images
        e = shap.DeepExplainer(smodel, (data['indicator'],A))
        # ...or pass tensors directly
        # e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
        shap_values = e.shap_values(data['indicator'][1:5])

        # plot the feature attributions
        shap.image_plot(shap_values, -data['indicator'][1:5])


