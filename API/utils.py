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
    X['P'] = 100 - X['N+A']
    results[self.N_col] = self._normalize(results[self.N_col])*X['N+A'].values.reshape(-1,1)
    results[self.P_col] = self._normalize(results[self.P_col])*X['P'].values.reshape(-1,1)

    return results.values
  
  @staticmethod
  def _normalize(x):
    return x/x.sum(axis=1).values.reshape(-1,1)

class transform23to54(object):
    def __init__(self):
        self.x_cols = y23.columns.tolist()
        self.y_cols = y54.columns.tolist()
        self.W = W
    
    def __call__(self,x):
        res = x.values@self.W
        return pd.DataFrame(res,columns=self.y_cols)

class Dual_net(nn.Module):
    def __init__(self):
        super(Dual_net,self).__init__()
        C_in = 4
        C_out = 3
        N_in = 54
        N_out = 54
        F_in = C_out+N_out
        F_out = C_out+N_out
        O_out = 3
        
        # build C,N,F
        self.C_net = self._build_C_net(C_in,C_out)
        self.N_net = self._build_N_net(N_in,N_out) 
        self.F_net = self._build_F_net(F_in,F_out)
        
        # build O_net
        for i in range(54):
            setattr(self,'O_net{}'.format(i+1),self._build_O_net(F_out,O_out))
        
        # initialize weight
        self.apply(self._init_weights)
            
    def forward(self,x):
        c,n = self._Fetch(x)
        c,n = self.C_net(c),self.N_net(n)
        f = torch.cat((c,n),dim=1)
        f = self.F_net(f)
        output = torch.tensor([]).cuda()
        for i in range(54):
            O_net = getattr(self,'O_net{}'.format(i+1))
            v = F.sigmoid(O_net(f))
            output = torch.cat((output,v),dim=1)
        return output
    
    @staticmethod
    def _Fetch(x):
        return x[:,:4],x[:,4:]
    
    @staticmethod
    def _build_C_net(input_shape,output_shape):
        net = torch.nn.Sequential(
            Linear(input_shape,128),
            ReLU(),
            Linear(128,output_shape))
        return net.cuda()
    
    @staticmethod
    def _build_N_net(input_shape,output_shape):
        net = torch.nn.Sequential(
            Linear(input_shape,128),
            ReLU(),
            Linear(128,output_shape))
        return net.cuda()
    
    @staticmethod
    def _build_F_net(input_shape,output_shape):
        net = torch.nn.Sequential(
            Linear(input_shape,128),
            ReLU(),
            Linear(128,output_shape))
        return net.cuda()
    
    @staticmethod
    def _build_O_net(input_shape,output_shape):
        net = torch.nn.Sequential(
            Linear(input_shape,128),
            ReLU(),
            Linear(128,output_shape))
        return net.cuda()
    
    @staticmethod
    def _init_weights(m):
        if hasattr(m,'weight'):
            torch.nn.init.xavier_uniform(m.weight)
        if hasattr(m,'bias'):
            m.bias.data.fill_(0)


class ANN_wrapper(object):
    def __init__(self,x_col,y_col,n_col,scaler,net):
        self.x_col = x_col
        self.y_col = y_col
        self.n_col = n_col
        self.scaler = scaler
        self.net = net
    
    def predict(self,x):
        x = self.scaler.transform(x)
        x = torch.tensor(x,dtype=torch.float).cuda()
        y = self.net(x).detach().cpu().numpy()
        y = pd.DataFrame(y,columns=self.y_col).apply(lambda x:round(x,12))
        y = self.normalize(y)
        return y
    
    def normalize(self,y):
        for i in self.n_col:
            le = 'Individual Component to Light End Split Factor_{}_Ratio'.format(i)
            hc = 'Individual Component to Heart Cut Split Factor_{}_Ratio'.format(i)
            he = 'Individual Component to Heavy End Split Factor_{}_Ratio'.format(i)
            col = [le,hc,he]
            y[col] = y[col].values / y[col].sum(axis=1).values.reshape(-1,1)
        return y

class transformer2(object):
    def __init__(self):
        # output columns
        self.le = get_col(df,'Light End Product Properties')[3:-1]
        self.hc = get_col(df,'Heart Cut Product Properties')[4:-1]
        self.he = get_col(df,'Heavy End Product Properties')[3:-1]
        
        # split factor columns
        self.le_sp = get_col(df,'Light End Split Factor')
        self.hc_sp = get_col(df,'Heart Cut Split Factor')
        self.he_sp = get_col(df,'Heavy End Split Factor')
    
    @staticmethod
    def _calculate_output(X,S,col_name):
        X, S = X.values, S.values
        F = np.diag(X@(S.T)).reshape(-1,1)
        Y = 100*(X*S)/(F)
        return pd.DataFrame(Y,columns=col_name)
    
    def __call__(self,xna,sp162):
        sle = sp162[self.le_sp] #SLE
        shc = sp162[self.hc_sp] #SHC
        she = sp162[self.he_sp] #SHE
        x_le = self._calculate_output(xna,sle,self.le) #XLE
        x_hc = self._calculate_output(xna,shc,self.hc) #XHC
        x_he = self._calculate_output(xna,she,self.he) #XHE
        return pd.concat([x_le,x_hc,x_he],axis=1)

class split_factor_transformer(object):
    def __init__(self):
        self.Xna_col = Xna_col
        self.Fna_col = Fna_col
        #=====================
        self.Sle_col = Sle_col
        self.Shc_col = Shc_col
        self.She_col = She_col
        #=====================
        self.Xle_col = Xle_col
        self.Xhc_col = Xhc_col
        self.Xhe_col = Xhe_col
        #=====================
        self.Fle_col = Fle_col
        self.Fhc_col = Fhc_col
        self.Fhe_col = Fhe_col
    
    def __call__(self,df):
        sle = self.calculate_sp(df,self.Xle_col,self.Fle_col,self.Sle_col)
        shc = self.calculate_sp(df,self.Xhc_col,self.Fhc_col,self.Shc_col)
        she = self.calculate_sp(df,self.Xhe_col,self.Fhe_col,self.She_col)
        #=================
        sle = sle.fillna(1)
        sle.iloc[:,-1] = 0
        #=================
        shc = shc.fillna(0)
        #=================
        she = she.fillna(0)
        she.iloc[:,-1] = 1
        return sle,shc,she
    
    def calculate_sp(self,df,x_col,f_col,s_col):
        Xna = df[self.Xna_col].values
        Fna = df[self.Fna_col].values.reshape(-1,1)
        Xw = df[x_col].values
        Fw = df[f_col].values.reshape(-1,1)
        return pd.DataFrame((Xw*Fw)/(Xna*Fna),columns=s_col)