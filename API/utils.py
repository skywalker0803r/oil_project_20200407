import pandas as pd
from sklearn.svm import SVR
from tqdm import tqdm_notebook as tqdm
import numpy as np
from sklearn.metrics import r2_score,mean_squared_error
from math import sqrt
from sklearn.preprocessing import StandardScaler
import warnings;warnings.simplefilter('ignore')
from sklearn.pipeline import Pipeline
import joblib
import pickle
import torch
import torchviz
from torch import nn
import torch.nn.functional as F
from torch import tensor
from torch.nn import Linear,ReLU,Sigmoid,Tanh
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import r2_score,mean_squared_error
from math import sqrt
import joblib
from torch.utils.tensorboard import SummaryWriter
import warnings;warnings.simplefilter('ignore')
from tqdm import tqdm_notebook as tqdm
import os
from sklearn.utils import shuffle

class custom_model(object):
  def __init__(self,x_cols,y_cols):
    self.x_cols = x_cols
    self.y_cols = y_cols
    self.N_col = ['C5N','C6N','C6A','C7N','C7A','C8N','C8A','C9N','C9A','C10N','C10A']
    self.P_col = ['C5NP','C5IP','C6NP','C6IP','C7NP','C7IP','C8NP','C8IP','C9NP','C9IP','C10NP','C10IP']
    self.model_23 = {}
    for y_name in y_cols:
      self.model_23[y_name] = Pipeline([('scaler',StandardScaler()),('reg',SVR(C=0.3))])
  
  def fit(self,X,y):
    for y_name in tqdm(self.y_cols):
      self.model_23[y_name].fit(X,y[y_name])
      y_pred = self.model_23[y_name].predict(X) 
      # Sequence prediction add y_pred to X 
      X.loc[:,y_name] = y_pred
    # recover X
    X = X[self.x_cols]
  
  def predict(self,data):
    X = data.copy()
    results = pd.DataFrame(index=[*range(len(X))],columns=self.y_cols)
    for y_name in self.y_cols:
      y_pred = self.model_23[y_name].predict(X)
      results.loc[:,y_name] = y_pred
      # Sequence prediction add y_pred to X 
      X.loc[:,y_name] = y_pred
    # recover X
    X = X[self.x_cols]
    
    # normalize depand on N+A and P
    results[self.N_col] = self._normalize(results[self.N_col])*X['N+A'].values.reshape(-1,1)
    results[self.P_col] = self._normalize(results[self.P_col])*X['P'].values.reshape(-1,1)

    return results.values
  
  @staticmethod
  def _normalize(x):
    return x/x.sum(axis=1).values.reshape(-1,1)

class transform23to54(object):
    def __init__(self):
        self.x_cols = y23 
        self.y_cols = y54 
        self.W = tr_matrix
    def __call__(self,x):
        res = x.values@self.W.values
        return pd.DataFrame(res,columns=self.y_cols)

class Dual_net(nn.Module):
    def __init__(self):
        super(Dual_net,self).__init__()
        a = 3
        b = 54
        c = a+b
        self.C_net = self._build_C_net(4,a)
        self.N_net = self._build_N_net(b,b) 
        self.F_net = self._build_F_net(c,c)
        # build O_net
        self.O_net1 = self._build_O_net(c,3)
        self.O_net2 = self._build_O_net(c,3)
        self.O_net3 = self._build_O_net(c,3)
        self.O_net4 = self._build_O_net(c,3)
        self.O_net5 = self._build_O_net(c,3)
        self.O_net6 = self._build_O_net(c,3)
        self.O_net7 = self._build_O_net(c,3)
        self.O_net8 = self._build_O_net(c,3)
        self.O_net9 = self._build_O_net(c,3)
        self.O_net10 = self._build_O_net(c,3)
        self.O_net11 = self._build_O_net(c,3)
        self.O_net12 = self._build_O_net(c,3)
        self.O_net13 = self._build_O_net(c,3)
        self.O_net14 = self._build_O_net(c,3)
        self.O_net15 = self._build_O_net(c,3)
        self.O_net16 = self._build_O_net(c,3)
        self.O_net17 = self._build_O_net(c,3)
        self.O_net18 = self._build_O_net(c,3)
        self.O_net19 = self._build_O_net(c,3)
        self.O_net20 = self._build_O_net(c,3)
        self.O_net21 = self._build_O_net(c,3)
        self.O_net22 = self._build_O_net(c,3)
        self.O_net23 = self._build_O_net(c,3)
        self.O_net24 = self._build_O_net(c,3)
        self.O_net25 = self._build_O_net(c,3)
        self.O_net26 = self._build_O_net(c,3)
        self.O_net27 = self._build_O_net(c,3)
        self.O_net28 = self._build_O_net(c,3)
        self.O_net29 = self._build_O_net(c,3)
        self.O_net30 = self._build_O_net(c,3)
        self.O_net31 = self._build_O_net(c,3)
        self.O_net32 = self._build_O_net(c,3)
        self.O_net33 = self._build_O_net(c,3)
        self.O_net34 = self._build_O_net(c,3)
        self.O_net35 = self._build_O_net(c,3)
        self.O_net36 = self._build_O_net(c,3)
        self.O_net37 = self._build_O_net(c,3)
        self.O_net38 = self._build_O_net(c,3)
        self.O_net39 = self._build_O_net(c,3)
        self.O_net40 = self._build_O_net(c,3)
        self.O_net41 = self._build_O_net(c,3)
        self.O_net42 = self._build_O_net(c,3)
        self.O_net43 = self._build_O_net(c,3)
        self.O_net44 = self._build_O_net(c,3)
        self.O_net45 = self._build_O_net(c,3)
        self.O_net46 = self._build_O_net(c,3)
        self.O_net47 = self._build_O_net(c,3)
        self.O_net48 = self._build_O_net(c,3)
        self.O_net49 = self._build_O_net(c,3)
        self.O_net50 = self._build_O_net(c,3)
        self.O_net51 = self._build_O_net(c,3)
        self.O_net52 = self._build_O_net(c,3)
        self.O_net53 = self._build_O_net(c,3)
        self.O_net54 = self._build_O_net(c,3)
        # O_nets list
        self.O_nets = [self.O_net1,self.O_net2,self.O_net3,self.O_net4,self.O_net5,
                       self.O_net6,self.O_net7,self.O_net8,self.O_net9,self.O_net10,
                       self.O_net11,self.O_net12,self.O_net13,self.O_net14,self.O_net15,
                       self.O_net16,self.O_net17,self.O_net18,self.O_net19,self.O_net20,
                       self.O_net21,self.O_net22,self.O_net23,self.O_net24,self.O_net25,
                       self.O_net26,self.O_net27,self.O_net28,self.O_net29,self.O_net30,
                       self.O_net31,self.O_net32,self.O_net33,self.O_net34,self.O_net35,
                       self.O_net36,self.O_net37,self.O_net38,self.O_net39,self.O_net40,
                       self.O_net41,self.O_net42,self.O_net43,self.O_net44,self.O_net45,
                       self.O_net46,self.O_net47,self.O_net48,self.O_net49,self.O_net50,
                       self.O_net51,self.O_net52,self.O_net53,self.O_net54]
        # initialize weight
        self.apply(self._init_weights)
            
    def forward(self,x):
        c,n = self._Fetch(x)
        c,n = self.C_net(c),self.N_net(n)
        f = torch.cat((c,n),dim=1)
        f = self.F_net(f)
        output = torch.tensor([]).cuda()
        for i in self.O_nets:
            v = F.softmax(i(f),dim=1)
            output = torch.cat((output,v),dim=1)
        return output
    
    @staticmethod
    def _Fetch(x):
        return x[:,:4],x[:,4:]
    
    @staticmethod
    def _build_C_net(input_shape,output_shape):
        net = torch.nn.Sequential(
            Linear(input_shape,128),ReLU(),
            Linear(128,output_shape))
        return net.cuda()
    
    @staticmethod
    def _build_N_net(input_shape,output_shape):
        net = torch.nn.Sequential(
            Linear(input_shape,128),ReLU(),
            Linear(128,output_shape))
        return net.cuda()
    
    @staticmethod
    def _build_F_net(input_shape,output_shape):
        net = torch.nn.Sequential(
            Linear(input_shape,128),ReLU(),
            Linear(128,output_shape))
        return net.cuda()
    
    @staticmethod
    def _build_O_net(input_shape,output_shape):
        net = torch.nn.Sequential(
            Linear(input_shape,128),ReLU(),
            Linear(128,output_shape))
        return net.cuda()
    
    @staticmethod
    def _init_weights(m):
        if hasattr(m,'weight'):
            torch.nn.init.xavier_uniform(m.weight)
        if hasattr(m,'bias'):
            m.bias.data.fill_(0)


class ANN_wrapper(object):
    def __init__(self,x_col,y_col,scaler,net):
        self.x_col = x_col
        self.y_col = y_col
        self.scaler = scaler
        self.net = net
    
    def predict(self,x):
        x = self.scaler.transform(x)
        x = torch.tensor(x,dtype=torch.float).cuda()
        y = self.net(x).detach().cpu().numpy()
        y = pd.DataFrame(y,columns=self.y_col)
        return y