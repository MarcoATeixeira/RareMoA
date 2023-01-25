#%%
import os
from typing import Callable, Union
import numpy as np
import pandas as pd
import joblib
from torch.utils.data import Dataset, DataLoader
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

#%%
def get_datasets(
    fold, 
    threshold = 100, 
    random_state = 23, 
    percentage = 10,
    return_classes = False,
    base_path: str = 'MoA Dataset\\Samples With Folds'
):

    '''
    Returns
    ---------

    dataset: dict

        dataset
        |
        L train
        |   L X
        |   L y
        |       L major
        |       L both
        |
        L eval
        |   L (...)
        |
        L test
            L (...)    
    '''
    dset = {
        k: {
            'X': [],
            'y': {
                'major': [],
                'both': []
            }
        } for k in ['train', 'eval', 'test']
    }

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
        dset[set_]['X'] = data.drop(
            ['cp_type', 'kfold'],
            axis = 'columns'
        ).astype(float)
        dset[set_]['y']['both'] = ys
        dset[set_]['y']['major'] = ys[[i for i in ys.columns if i not in minor_classes]]

    # Build the eval sets
    mskf = MultilabelStratifiedKFold(
        n_splits = percentage, 
        shuffle = True, 
        random_state = random_state
    )
    train_idx, eval_idx = next(
            mskf.split(
                dset['train']['X'], 
                dset['train']['y']['both']
            )
        )

    dset['eval']['X'] = dset['train']['X'].iloc[eval_idx,:]
    dset['eval']['y']['both'] = dset['train']['y']['both'].iloc[eval_idx,:]
    dset['eval']['y']['major'] = dset['train']['y']['major'].iloc[eval_idx,:]
    dset['train']['X'] = dset['train']['X'].iloc[train_idx,:]
    dset['train']['y']['both'] = dset['train']['y']['both'].iloc[train_idx,:]
    dset['train']['y']['major'] = dset['train']['y']['major'].iloc[train_idx,:]

    if return_classes: return dset, minor_classes
    else: return dset
        


# %%
