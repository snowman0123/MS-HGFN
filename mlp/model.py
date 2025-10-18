import numpy as np
from torch import nn
import torch
from torch.nn import Module, Parameter
import math
import torch.nn.functional as F
import warnings

from Prediction import Predict

warnings.filterwarnings("ignore")


class Model(nn.Module):
    def __init__(self, in_size, hid_size, time_length, task):
        super(Model, self).__init__()

        self.in_size = in_size
        self.hid_size = hid_size
        self.fc = nn.Linear(in_size * time_length, hid_size)
        self.pred = Predict(hid_size, task)

    def forward(self, x):
        b, t, n, d = x.shape
        x=torch.transpose(x,1,2)

        x = x.reshape(b, n,-1)
        x = self.fc(x)
        out = self.pred(x)
        return out

class EarlyStopping:
    def __init__(self, args,device,patience=5, verbose=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_ls = None
        self.best_model = Model(args.in_size,args.hidden_feat,args.time_length,args.task).to(device)
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