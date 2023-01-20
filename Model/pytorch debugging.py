#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 21:29:58 2023

@author: brechtl

This is just a file I used to debug my PyTorch model and tried some parameter tuning.
The final model can be found in the Jupyter Notebook "4. PyTorch"

"""

from numpy import vstack
from numpy import sqrt
from pandas import read_csv
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import ReLU
from torch.nn import Module
from torch.optim import SGD
from torch.nn import MSELoss
from torch.nn.init import xavier_uniform_
import torch as T
import pandas as pd
import numpy as np
from torchmetrics import R2Score

# dataset definition
class CSVDataset(Dataset):
    def __init__(self, path):
        # load the csv file as a dataframe
        df = read_csv(path)
        # store the inputs and outputs
        self.X = df.drop('Mean Scale Score', axis=1)
        self.y = df['Mean Scale Score']
        # ensure target has the right shape
        self.y = self.y.values.reshape((-1, 1))
        
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)
        # self.y = scaler.fit_transform(self.y)
        
        self.X = T.tensor(self.X.astype(np.float32))
        self.y = T.tensor(self.y.astype(np.float32))
        
        # print(self.X)
        
        # self.X = self.X[:200]
        # self.y = self.y[:200]
        
        print(self.X[2])
        print(self.X[10])
        print(self.y[2])
        print(self.y[10])
        
        

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)
 
    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]
 
    # get indexes for train and test rows
    def get_splits(self, n_test=0.33):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])
    
# model definition
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 1)
        xavier_uniform_(self.hidden1.weight)
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(10, 10)
        xavier_uniform_(self.hidden2.weight)
        self.act2 = ReLU()
        # third hidden layer and output
        self.hidden3 = Linear(10, 1)
        xavier_uniform_(self.hidden3.weight)
 
    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        # X = self.act1(X)
        # # second hidden layer
        # X = self.hidden2(X)
        # X = self.act2(X)
        # # third hidden layer and output
        # X = self.hidden3(X)
        return X
    
    
# prepare the dataset
def prepare_data(path):
    # load the dataset
    dataset = CSVDataset(path)
    # calculate split
    train, test = dataset.get_splits()
    
    # scaler = StandardScaler()
    # train = scaler.fit_transform(train)
    # test = scaler.transform(test)
    
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=80, shuffle=True)
    test_dl = DataLoader(test, batch_size=120, shuffle=False)
    return train_dl, test_dl
 
# train the model
def train_model(train_dl, model):
    # define the optimization
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.08)
    # enumerate epochs
    for epoch in range(40):
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
        print('epoch {}, MSE: {}'.format(epoch, loss))
 
# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # store
        predictions.append(yhat)
        actuals.append(actual)
        
        predictions, actuals = vstack(predictions), vstack(actuals)
        
        print(r2_score(actuals, predictions))
        
        # calculate mse
        mse = mean_squared_error(actuals, predictions)
        return mse
    
 
# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat



path = "/home/brechtl/OneDrive/Lessen/AI Frameworks/Project/Data/Math_Test_Results_Cleaned.csv"
train_dl, test_dl = prepare_data(path)

# define the network
model = MLP(14)

# train the model
train_model(train_dl, model)
# evaluate the model
mse = evaluate_model(test_dl, model)
print('MSE: %.3f, RMSE: %.3f' % (mse, sqrt(mse)))

row1 = [-1.3031, -0.5366, -0.8207, -0.6752, -0.7587, -0.7223, -0.6078, -0.4697, 2.1286, -0.6750, -0.8107, -1.1995, -1.5261, -1.1210]
row2 = [-0.6986, -0.0346, -0.7993, -0.4013,  0.0282, -0.4451,  0.7323, -0.7199, 0.1748, -0.6750, -0.8269, -1.1995, -1.5261, -1.1210]
yhat = predict(row1, model)
print('Predicted: %.3f' % yhat)
print(668)

yhat = predict(row2, model)
print('Predicted: %.3f' % yhat)
print(655)




