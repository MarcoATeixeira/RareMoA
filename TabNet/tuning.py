# -*- coding: utf-8 -*-
"""After DACO.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1C4jy6cjex7aLiYDbtXypNyU0aZ9noyNA
"""

#%% Install packages
!pip install pytorch-tabnet
!pip install iterative-stratification

#%% Imports
from google.colab import drive
import zipfile
from pytorch_tabnet.tab_model import TabNetRegressor
import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest
import matplotlib.pyplot as plt
import pickle as pk
import joblib
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
import torch

#%% Importar drive
drive.mount('/content/drive')

#%% TabNet Parameters
MAX_EPOCH=200
tabnet_params = dict(n_d=25, n_a=25, n_steps=1, gamma=1.3,
                     lambda_sparse=0, optimizer_fn=torch.optim.Adam,
                     optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                     mask_type='entmax',
                     scheduler_params=dict(mode="min",
                                           patience=5,
                                           min_lr=1e-5,
                                           factor=0.9,),
                     scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                     verbose=10,
                     )

#%% Log-Loss
from pytorch_tabnet.metrics import Metric

class LogitsLogLoss(Metric):
    """
    LogLoss with sigmoid applied
    """

    def __init__(self):
        self._name = "logits_ll"
        self._maximize = False

    def __call__(self, y_true, y_pred):
        """
        Compute LogLoss of predictions.

        Parameters
        ----------
        y_true: np.ndarray
            Target matrix or vector
        y_score: np.ndarray
            Score matrix or vector

        Returns
        -------
            float
            LogLoss of predictions vs targets.
        """
        logits = 1 / (1 + np.exp(-y_pred))
        aux = (1-y_true)*np.log(1-logits+1e-15) + y_true*np.log(logits+1e-15)
        return np.mean(-aux)

#%% Tuning n_steps
def run_tuning(features, labels, seed):
    hyper=[1,2]
    minLoss=100
    minI=10

    for i in hyper:
        print("i = %2d" % (i))

        losses=np.zeros((5,1))
        mskf = MultilabelStratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for fold_nb, (train_idx, val_idx) in enumerate(mskf.split(features, labels)):
            print("VAL FOLD N.%2d" % (fold_nb))

            x_train = features.loc[features.index[train_idx], :]
            y_train = labels.loc[features.index[train_idx], :]

            x_val = features.loc[features.index[val_idx], :]
            y_val = labels.loc[features.index[val_idx], :]

            preds = run_training(x_train, y_train, x_val, y_val, i)
            logloss = LogitsLogLoss()
            losses[fold_nb] = logloss(y_val.values, preds)
            print("i = %2d, loss = %5.5f" % (i, losses[fold_nb]))

        loss = np.mean(losses)
        if (loss < minLoss):
            minLoss = loss
            minI = i

    return minI

def run_training(features, labels, val_features, val_labels, hyper):

    x_train_first = features.to_numpy().astype(float)
    y_train_first = labels.loc[:, pd.concat((labels,val_labels),axis=0).sum(axis=0) > 100].to_numpy().astype(float)
    x_val_first = val_features.to_numpy().astype(float)
    y_val_first = val_labels.loc[:, pd.concat((labels,val_labels),axis=0).sum(axis=0) > 100].to_numpy().astype(float)

    tabnet_params = dict(n_d=25, n_a=25, n_steps=hyper, gamma=1.3,
                     lambda_sparse=0, optimizer_fn=torch.optim.Adam,
                     optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                     mask_type='entmax',
                     scheduler_params=dict(mode="min", patience=5, min_lr=1e-5, factor=0.9,),
                     scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                     verbose=10,)

    model_first = TabNetRegressor(**tabnet_params)
    model_first.fit(X_train=x_train_first,
              y_train=y_train_first,
              eval_set=[(x_val_first, y_val_first)],
              eval_name=["val"],
              eval_metric=["logits_ll"],
              max_epochs=MAX_EPOCH,
              patience=20, batch_size=1024, virtual_batch_size=128,
              num_workers=1, drop_last=False,
              # use binary cross entropy as this is not a regression problem
              loss_fn=torch.nn.functional.binary_cross_entropy_with_logits)
    
    print(x_train_first.dtype)
    preds_train_first = model_first.predict(x_train_first)
    preds_val_first = model_first.predict(x_val_first)


    x_train_second = np.concatenate((x_train_first, preds_train_first),axis=1)
    y_train_second = labels.values
    x_val_second = np.concatenate((x_val_first, preds_val_first),axis=1)
    y_val_second = val_labels.values

    tabnet_params = dict(n_d=25, n_a=25, n_steps=hyper, gamma=1.3,
                     lambda_sparse=0, optimizer_fn=torch.optim.Adam,
                     optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                     mask_type='entmax',
                     scheduler_params=dict(mode="min", patience=5, min_lr=1e-5, factor=0.9,),
                     scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                     verbose=10,)

    model_second = TabNetRegressor(**tabnet_params)
    model_second.fit(X_train=x_train_second,
              y_train=y_train_second,
              eval_set=[(x_val_second,y_val_second)],
              eval_name=["val"],
              eval_metric=["logits_ll"],
              max_epochs=MAX_EPOCH,
              patience=20, batch_size=1024, virtual_batch_size=128,
              num_workers=1, drop_last=False,
              # use binary cross entropy as this is not a regression problem
              loss_fn=torch.nn.functional.binary_cross_entropy_with_logits)
    
    preds_val_second = model_second.predict(x_val_second)

    return preds_val_second

RAN=23

folds = np.array([0,1,2,3,4])
fold = 4
for count, f in enumerate([j for j in folds if j != fold]):

  if (count != 0):
    data = joblib.load("/content/drive/MyDrive/AfterDACO/samples_fold_" + str(f) + ".pkl")
    x_train = pd.concat((x_train, data), axis=0)
    data = joblib.load("/content/drive/MyDrive/AfterDACO/labels_fold_" + str(f) + ".pkl")
    y_train = pd.concat((y_train, data), axis=0)
  else:
    x_train = joblib.load("/content/drive/MyDrive/AfterDACO/samples_fold_" + str(f) + ".pkl")
    y_train = joblib.load("/content/drive/MyDrive/AfterDACO/labels_fold_" + str(f) + ".pkl")

x_train = x_train.drop(['cp_type','kfold'],axis=1)
x_train['cp_dose'][x_train['cp_dose']=='D1'] = 1
x_train['cp_dose'][x_train['cp_dose']=='D2'] = 2

x_test = joblib.load("/content/drive/MyDrive/AfterDACO/samples_fold_" + str(fold) + ".pkl")
y_test = joblib.load("/content/drive/MyDrive/AfterDACO/labels_fold_" + str(fold) + ".pkl")

print("TEST FOLD N.%2d" % (fold))

i = run_tuning(x_train, y_train, RAN)

#%% Tuning different steps
def run_tuning(features, labels, seed):
    hyper=[1,2]
    minLoss=100
    minI=10

    for i in hyper:
        print("i = %2d" % (i))

        losses=np.zeros((5,1))
        mskf = MultilabelStratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for fold_nb, (train_idx, val_idx) in enumerate(mskf.split(features, labels)):
            print("VAL FOLD N.%2d" % (fold_nb))

            x_train = features.loc[features.index[train_idx], :]
            y_train = labels.loc[features.index[train_idx], :]

            x_val = features.loc[features.index[val_idx], :]
            y_val = labels.loc[features.index[val_idx], :]

            preds = run_training(x_train, y_train, x_val, y_val, i)
            logloss = LogitsLogLoss()
            losses[fold_nb] = logloss(y_val.values, preds)
            print("i = %2d, loss = %5.5f" % (i, losses[fold_nb]))

        loss = np.mean(losses)
        if (loss < minLoss):
            minLoss = loss
            minI = i

    return minI

def run_training(features, labels, val_features, val_labels, hyper):

    x_train_first = features.to_numpy().astype(float)
    y_train_first = labels.loc[:, pd.concat((labels,val_labels),axis=0).sum(axis=0) > 100].to_numpy().astype(float)
    x_val_first = val_features.to_numpy().astype(float)
    y_val_first = val_labels.loc[:, pd.concat((labels,val_labels),axis=0).sum(axis=0) > 100].to_numpy().astype(float)

    tabnet_params = dict(n_d=25, n_a=25, n_steps=1, gamma=1.3,
                     lambda_sparse=0, optimizer_fn=torch.optim.Adam,
                     optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                     mask_type='entmax',
                     scheduler_params=dict(mode="min", patience=5, min_lr=1e-5, factor=0.9,),
                     scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                     verbose=10,)

    model_first = TabNetRegressor(**tabnet_params)
    model_first.fit(X_train=x_train_first,
              y_train=y_train_first,
              eval_set=[(x_val_first, y_val_first)],
              eval_name=["val"],
              eval_metric=["logits_ll"],
              max_epochs=MAX_EPOCH,
              patience=20, batch_size=1024, virtual_batch_size=128,
              num_workers=1, drop_last=False,
              # use binary cross entropy as this is not a regression problem
              loss_fn=torch.nn.functional.binary_cross_entropy_with_logits)
    
    print(x_train_first.dtype)
    preds_train_first = model_first.predict(x_train_first)
    preds_val_first = model_first.predict(x_val_first)


    x_train_second = np.concatenate((x_train_first, preds_train_first),axis=1)
    y_train_second = labels.values
    x_val_second = np.concatenate((x_val_first, preds_val_first),axis=1)
    y_val_second = val_labels.values

    tabnet_params = dict(n_d=25, n_a=25, n_steps=hyper, gamma=1.3,
                     lambda_sparse=0, optimizer_fn=torch.optim.Adam,
                     optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                     mask_type='entmax',
                     scheduler_params=dict(mode="min", patience=5, min_lr=1e-5, factor=0.9,),
                     scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                     verbose=10,)

    model_second = TabNetRegressor(**tabnet_params)
    model_second.fit(X_train=x_train_second,
              y_train=y_train_second,
              eval_set=[(x_val_second,y_val_second)],
              eval_name=["val"],
              eval_metric=["logits_ll"],
              max_epochs=MAX_EPOCH,
              patience=20, batch_size=1024, virtual_batch_size=128,
              num_workers=1, drop_last=False,
              # use binary cross entropy as this is not a regression problem
              loss_fn=torch.nn.functional.binary_cross_entropy_with_logits)
    
    preds_val_second = model_second.predict(x_val_second)

    return preds_val_second

RAN=23

folds = np.array([0,1,2,3,4])
fold = 4
for count, f in enumerate([j for j in folds if j != fold]):

  if (count != 0):
    data = joblib.load("/content/drive/MyDrive/AfterDACO/samples_fold_" + str(f) + ".pkl")
    x_train = pd.concat((x_train, data), axis=0)
    data = joblib.load("/content/drive/MyDrive/AfterDACO/labels_fold_" + str(f) + ".pkl")
    y_train = pd.concat((y_train, data), axis=0)
  else:
    x_train = joblib.load("/content/drive/MyDrive/AfterDACO/samples_fold_" + str(f) + ".pkl")
    y_train = joblib.load("/content/drive/MyDrive/AfterDACO/labels_fold_" + str(f) + ".pkl")

x_train = x_train.drop(['cp_type','kfold'],axis=1)
x_train['cp_dose'][x_train['cp_dose']=='D1'] = 1
x_train['cp_dose'][x_train['cp_dose']=='D2'] = 2

x_test = joblib.load("/content/drive/MyDrive/AfterDACO/samples_fold_" + str(fold) + ".pkl")
y_test = joblib.load("/content/drive/MyDrive/AfterDACO/labels_fold_" + str(fold) + ".pkl")

print("TEST FOLD N.%2d" % (fold))

i = run_tuning(x_train, y_train, RAN)

#%% Get result
i