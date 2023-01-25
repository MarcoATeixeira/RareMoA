#%%
import os
from typing import Callable, Union
import numpy as np
import pandas as pd
import joblib
from torch.utils.data import Dataset, DataLoader
import torch
#%%
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

class MoADataset(Dataset):
    """
    MoA Dataset.
    """

    def __init__(self, path, fold, train=True):
        self.path = path
        self.fold = fold
        self.train = train
        self.data = joblib.load(os.path.join(self.path, f'samples_fold_{0}.pkl'))
        self.labels = joblib.load(os.path.join(self.path, f'labels_fold_{0}.pkl'))
        for i in range(1,5):
            self.data = pd.concat([self.data, joblib.load(os.path.join(self.path, f'samples_fold_{i}.pkl'))])
            self.labels = pd.concat([self.labels, joblib.load(os.path.join(self.path, f'labels_fold_{i}.pkl'))])

        self.features = [name.replace('-','_') for name in list(self.data.drop(['cp_type', 'kfold'], axis=1).columns)]
        if self.train:
            self.x_train = torch.tensor(self.data[self.data['kfold']!=self.fold].drop(['kfold', 'cp_type'], axis=1).to_numpy(dtype=np.double), dtype=torch.float)
            self.y_train = torch.tensor(self.labels[self.data['kfold']!=self.fold].to_numpy(dtype=int), dtype=torch.float)
        else:
            self.x_test = torch.tensor(self.data[self.data['kfold']==self.fold].drop(['kfold', 'cp_type'], axis=1).to_numpy(dtype=np.double), dtype=torch.float)
            self.y_test = torch.tensor(self.labels[self.data['kfold']==self.fold].to_numpy(dtype=int), dtype=torch.float)

    def __len__(self):
        if self.train:
            return len(self.data[self.data['kfold']!=self.fold].drop(['kfold', 'cp_type'], axis=1))
        else:
            return len(self.data[self.data['kfold']==self.fold].drop(['kfold', 'cp_type'], axis=1))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.train:
            return self.x_train[idx,:], self.y_train[idx,:]
        else:
            return self.x_test[idx,:], self.y_test[idx,:]

    def get_dataframes(self):
        if self.train:
            return self.data[self.data['kfold']!=self.fold], self.labels[self.data['kfold']!=self.fold]
        else:
            return self.data[self.data['kfold']==self.fold], self.labels[self.data['kfold']==self.fold]

    def major_minor(self, threshold=20):
        '''
        Returns a dataframe with all majority classes encoded as 0 and all minority
        classes encoded as 1. Minority classes are those with less than a threshold
        number of instances. If threshold is in ]0, 1[, threshold % of the classes 
        with less support are selected as minority classes.
        '''
        self.support = self.labels.sum(axis=0)
        if threshold < 1:
            threshold = self.support[self.labels.columns[np.argsort(self.support.to_numpy())][int(self.support.shape[0]*threshold)]]
        self.major_minor_ = self.support < threshold
        self.threshold = threshold
        return self.major_minor_

    def get_major(self):
        return MoASubset(self.data, self.labels, fold=self.fold, mask=self.major_minor_==0, train=self.train)
    def get_minor(self):
        return MoASubset(self.data, self.labels, fold=self.fold, mask=self.major_minor_==1, train=self.train)

    def get_dataset_as_pandas(self):
        '''
        Returns the data and labels as Pandas DataFrames,
        instead of Dataset objects.
        '''
        X, y = self.get_dataframes()
        X = X.drop(['cp_type', 'kfold'], axis = 'columns').astype(float)
        return X, y.astype(int)


class MoASubset(MoADataset):
    '''
    Class for building a set by selecting a subset of an existing dataset.
    '''
    def __init__(
        self,
        X,
        y,
        mask,
        fold,
        train=True        
    ):

        self.labels = y
        self.data = X[np.any(y[y.columns[mask]], axis=1)]
        self.labels = self.labels.loc[self.data.index]
        self.fold = fold
        self.train = train
        self.features = [name.replace('-','_') for name in list(self.data.drop(['cp_type', 'kfold'], axis=1).columns)]
        if self.train:
            self.x_train = torch.tensor(self.data[self.data['kfold']!=self.fold].drop(['kfold', 'cp_type'], axis=1).to_numpy(dtype=np.double), dtype=torch.float)
            self.y_train = torch.tensor(self.labels[self.data['kfold']!=self.fold].to_numpy(dtype=int), dtype=torch.float)
        else:
            self.x_test = torch.tensor(self.data[self.data['kfold']==self.fold].drop(['kfold', 'cp_type'], axis=1).to_numpy(dtype=np.double), dtype=torch.float)
            self.y_test = torch.tensor(self.labels[self.data['kfold']==self.fold].to_numpy(dtype=int), dtype=torch.float)

'''
def get_datasets(fold, threshold=100, random_state=23, percentage=10):
    mskf = MultilabelStratifiedKFold(n_splits=percentage, shuffle=True, random_state=random_state)
    train = MoADataset('C:\\Users\\asus\\Desktop\\MIB\\4º Ano\\1º Semestre\\DACO\\MoAPrediction\\MoA Dataset\\Samples With Folds', fold=fold)
    test = MoADataset('C:\\Users\\asus\\Desktop\\MIB\\4º Ano\\1º Semestre\\DACO\\MoAPrediction\\MoA Dataset\\Samples With Folds', fold=fold, train=False)
    train.major_minor(threshold = threshold)
    test.major_minor(threshold = threshold)
    # Class -> Set -> X or y
    dset = {
        'minor': {
            'test': {k: v for k, v in zip(
                ['X', 'y'],
                test.get_minor().get_dataset_as_pandas()
            )},
            'train': {k: v for k, v in zip(
                ['X', 'y'],
                train.get_minor().get_dataset_as_pandas()
            )},
        },
        'major': {
            'test': {k: v for k, v in zip(
                ['X', 'y'],
                test.get_major().get_dataset_as_pandas()
            )},
            'train': {k: v for k, v in zip(
                ['X', 'y'],
                train.get_major().get_dataset_as_pandas()
            )},
        }
    }
    for class_ in ['major', 'minor']:
        if len(dset[class_]['train']['X']) > 0:
            train_idx, eval_idx = next(mskf.split(dset[class_]['train']['X'], dset[class_]['train']['y']))
            dset[class_]['eval'] = {
                'X': dset[class_]['train']['X'].iloc[eval_idx,:],
                'y': dset[class_]['train']['y'].iloc[eval_idx,:]
            }
            for k in ['X', 'y']: 
                dset[class_]['train'][k] = dset[class_]['train'][k].iloc[train_idx,:]

    return dset
'''
#%%
def get_datasets(
    fold, 
    threshold = 100, 
    random_state = 23, 
    percentage = 10,
    return_classes = False
):

    dset = {
        k: {
            kk : {
                'X': [],
                'y': []
            } for kk in ['train', 'eval', 'test']
        } for k in ['major', 'minor']
    }

    base_path = 'C:\\Users\\marco\\Desktop\\MIB\\4º Ano\\1º Semestre\\DACO\\MoAPrediction\\MoA Dataset\\Samples With Folds'
    # Load all labels, for all folds
    labels = pd.concat(
        [
            joblib.load(
                os.path.join(
                    base_path,
                    f'labels_fold_{i}.pkl'
                )
            )
            for i in range(5)
        ]
    )
    # Get the number of instances of each class
    support = np.sum(labels, axis = 0)
    # Which are the minor classes
    minor_classes = list(support.index[support < threshold])
    
    # Build the test set
    X_test = joblib.load(
        os.path.join(
            base_path,
            f'samples_fold_{fold}.pkl'
        )
    )
    y_test = joblib.load(
        os.path.join(
            base_path,
            f'labels_fold_{fold}.pkl'
        )
    )
    # Get all training set samples, regardless of class
    X = pd.concat(
        [
            joblib.load(
                os.path.join(
                    base_path,
                    f'samples_fold_{i}.pkl'
                )
            )
            for i in range(5) if i != fold
        ]
    )
    y = pd.concat(
        [
            joblib.load(
                os.path.join(
                    base_path,
                    f'labels_fold_{i}.pkl'
                )
            )
            for i in range(5) if i != fold
        ]
    )
    
    for set_, data, ys in zip(['test', 'train'], [X_test, X], [y_test, y]):

        # Samples to include in the minor set. All with at least one minor class
        # and all control samples
        dset['minor'][set_]['X'] = data[
            np.logical_or(
                np.any(
                    ys[minor_classes],
                    axis = 1
                ),
                data['cp_type'] == 'ctl_vehicle'
            )
        ].drop(
            ['cp_type', 'kfold'],
            axis = 'columns'
        ).astype(float)
        dset['minor'][set_]['y'] = ys.loc[dset['minor'][set_]['X'].index]

        # The major set is made up of the samples not included in the minor set
        dset['major'][set_]['X'] = data.drop(
            dset['minor'][set_]['X'].index
        ).drop(
            ['cp_type', 'kfold'],
            axis = 'columns'
        ).astype(float)
        dset['major'][set_]['y'] = ys.drop(dset['minor'][set_]['y'].index)

    # Build the eval sets
    mskf = MultilabelStratifiedKFold(
        n_splits = percentage, 
        shuffle = True, 
        random_state = random_state
    )
    for class_ in ['major', 'minor']:
        train_idx, eval_idx = next(
            mskf.split(
                dset[class_]['train']['X'], 
                dset[class_]['train']['y']
            )
        )
        for k in ['X', 'y']: 
            dset[class_]['eval'][k] = dset[class_]['train'][k].iloc[eval_idx,:]
            dset[class_]['train'][k] = dset[class_]['train'][k].iloc[train_idx,:]

    if return_classes: return dset, minor_classes
    else: return dset
        


# %%
