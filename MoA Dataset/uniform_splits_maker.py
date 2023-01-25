import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import random
import os
import joblib
N_FOLDS=5
RAN=23


class MoADataset:
    
    path_to_features = ''
    path_to_labels = ''
    path_to_nonscored = ''
    features = []
    labels = []
    nonscored_labels = []
    read_nonscored = True
    
    def __init__(self, path, read_nonscored = True):
        '''
        MoA Dataset handler. Reads feature and label data (both scored and non-
        scored). The dataset file names should be left as-is after download.

        Parameters
        ----------
        path : String.
            Path to MoA dataset directory.
        get_nonscored : Boolean, optional
            Read non-scored labels. The default is True.

        Returns
        -------
        None.

        '''
        
        self.path_to_features = os.path.join(path, 'train_features.csv')
        self.path_to_labels = os.path.join(path, 'train_targets_scored.csv')
        self.path_to_nonscored = os.path.join(path, 'train_targets_nonscored.csv')
        self.read_nonscored = read_nonscored
        
        self.get_features()
        self.get_labels()
        if self.read_nonscored: self.get_nonscored()
        
    def get_features(self, drop = None, axis = None):
        '''
        Returns feature data for the MoA dataset.

        Parameters
        ----------
        drop : List or None, optional
            List of columns or lines to drop. The default is None.
        axis : String, int or None, optional
            Axis in which the items to drop will be searched. Must be 0 or 1,
            or 'columns' or 'index'. The default is None.

        Returns
        -------
        Features : Pandas DataFrame.

        '''
        
        self.features = pd.read_csv(self.path_to_features, delimiter=',', index_col='sig_id')
        self.features['cp_dose'][self.features['cp_dose']=='D1'] = 1
        self.features['cp_dose'][self.features['cp_dose']=='D2'] = 2
        
        if drop is not None and axis is not None: 
            self.features = self.features.drop(drop, axis=axis)
            
        return self.features
            
    def get_labels(self, drop = None, axis = None):
        '''
        Returns label data for the MoA dataset.

        Parameters
        ----------
        drop : List or None, optional
            List of columns or lines to drop. The default is None.
        axis : String, int or None, optional
            Axis in which the items to drop will be searched. Must be 0 or 1,
            or 'columns' or 'index'. The default is None.

        Returns
        -------
        Labels : Pandas DataFrame.

        '''
        
        self.labels = pd.read_csv(self.path_to_labels, delimiter=',', index_col='sig_id')

        if drop is not None and axis is not None: 
            self.labels = self.labels.drop(drop, axis=axis)
            
        return self.labels
            
    def get_nonscored(self, drop = None, axis = None):
        
        '''
        Returns non-scored label data for the MoA dataset.

        Parameters
        ----------
        drop : List or None, optional
            List of columns or lines to drop. The default is None.
        axis : String, int or None, optional
            Axis in which the items to drop will be searched. Must be 0 or 1,
            or 'columns' or 'index'. The default is None.

        Returns
        -------
        Non-scored labels : Pandas DataFrame.

        '''
        
        self.nonscored_labels = pd.read_csv(self.path_to_nonscored, delimiter=',', index_col='sig_id')

        if drop is not None and axis is not None: 
            self.nonscored_labels = self.nonscored_labels.drop(drop, axis=axis)
            
        return self.nonscored_labels
    
    def get_data(self):
        '''
        Returns all data from the dataset, as a list.

        Returns
        -------
        List
            List of features, labels and non-scored labels (if read_nonscroed =
            True).

        '''
        
        if self.read_nonscored:
            return self.features, self.labels, self.nonscored_labels
        else:
            return self.features, self.labels
        
    def split_controls(self, drop=True):
        '''
        Splits the dataset into control perturbations and non-control samples.

        Parameters
        ----------
        drop : Boolean, optional
            If true, drops the 'cp_type' column from the dataset. The default is True.

        Returns
        -------
        samples : Pandas DataFrame
            Non-control sample data.
        ctrl : Pandas DataFrame
            Control data.

        '''
        
        ctrl_idx = self.features[self.features['cp_type'] == 'ctl_vehicle'].index
        samples = self.features.drop(ctrl_idx)
        sample_labels = self.labels.drop(ctrl_idx)
        ctrl = self.features.loc[ctrl_idx]
        ctrl_labels = self.labels.loc[ctrl_idx]
        
        if drop:
            samples.drop(['cp_type'], axis='columns', inplace=True)
            ctrl.drop(['cp_type'], axis='columns', inplace=True)
        
        if self.read_nonscored:
            sample_nonscored = self.nonscored_labels.drop(ctrl_idx)
            ctrl_nonscored = self.nonscored_labels.loc[ctrl_idx]
            return samples, sample_labels, sample_nonscored, ctrl, ctrl_labels, ctrl_nonscored
        else:
            return samples, sample_labels, sample_nonscored, ctrl, ctrl_labels, ctrl_nonscored
            
    def split_cell_gene_data(self, X=None):
        '''
        Splits the dataset into cell and gene data.

        Parameters
        ----------
        X : None or Pandas DataFrame, optional
            Dataset to split. If None, uses the original dataset, as is, from
            the csv file.

        Returns
        -------
        gene_set : Pandas DataFrame
            Gene dataset.
        cell_set : Pandas DataFrame
            Cell dataset.

        '''
        
        if X is None: X = self.features
        
        gene_columns = X.columns[X.columns.str.startswith('g-')]
        cell_columns = X.columns[X.columns.str.startswith('c-')]
        info_columns = list(set(X.columns).difference(gene_columns+cell_columns))
        
        return X[info_columns+gene_columns], X[info_columns+cell_columns]


def make_uniform_splits():
    '''
    Splits the data in a deterministic way.
    '''
    random.seed(RAN)
    
    data_path = os.path.join('MoA Dataset') # Edit this
    dataset = MoADataset(data_path)
    samples, sample_labels, y_extra = dataset.get_data()

    train_drug = pd.read_csv(os.path.join(data_path, 'train_drug.csv'))

    # get drug ids
    labels_sig = list(sample_labels.index)
    label_cols = sample_labels.columns
    train_drug = train_drug.loc[[i in labels_sig for i in train_drug['sig_id']]].reset_index(drop=True)
    # merge sample_labels with train_drug -> get a dataframe with sig_id, labels and drug used
    sample_labels_drugid = sample_labels.merge(train_drug, on='sig_id', how='left')

    # Locate drugs
    vc = train_drug.drug_id.value_counts()
    vc1 = vc.loc[vc <= 19].index # drug id of drugs that appear <= 19x in the train set
    vc2 = vc.loc[vc > 19].index # drug id of drugs that appear > 19x in the train set    

    # Kfold - leave drug out
    dct1 = {}; dct2 = {}

    skf = MultilabelStratifiedKFold(n_splits = N_FOLDS) 
    tmp = sample_labels_drugid.groupby('drug_id')[label_cols].mean().loc[vc1] # selecionar apenas as drugs do conjunto vc1
    # tmp.index retorna ids das drogas
    tmp_idx = tmp.index.tolist()
    tmp_idx.sort()
    tmp_idx2 = random.sample(tmp_idx,len(tmp_idx))  
    tmp = tmp.loc[tmp_idx2] # random order

    #distribuir as drugs que aparecem <= 19x por folds
    for fold,(idxT,idxV) in enumerate(skf.split(tmp,tmp[label_cols])):
      dd = {k:fold for k in tmp.index[idxV].values} 
      dct1.update(dd)

    skf = MultilabelStratifiedKFold(n_splits = N_FOLDS) 
    tmp = sample_labels_drugid.loc[sample_labels_drugid.drug_id.isin(vc2)].reset_index(drop = True) # selecionar apenas as drugs do conjunto vc2
    #tmp.sig_id retorna ids das amostras
    tmp_idx = tmp.index.tolist()
    tmp_idx.sort()
    tmp_idx2 = random.sample(tmp_idx,len(tmp_idx))  
    tmp = tmp.loc[tmp_idx2] # random order

    #distribuir amostras cujas drugs aparecem >19x por folds
    for fold,(idxT,idxV) in enumerate(skf.split(tmp,tmp[label_cols])):
      dd = {k:fold for k in tmp.sig_id[idxV].values}
      dct2.update(dd)

    # colocar coluna kfold em frente a cada amostra:
    sample_labels_drugid['kfold'] = sample_labels_drugid.drug_id.map(dct1)
    sample_labels_drugid.loc[sample_labels_drugid.kfold.isna(),'kfold'] = sample_labels_drugid.loc[sample_labels_drugid.kfold.isna(),'sig_id'].map(dct2)
    sample_labels_drugid.kfold = sample_labels_drugid.kfold.astype(int)

    # adicionar coluna ao samples com o fold de cada amostra
    sample_labels_drugid.index=sample_labels_drugid['sig_id']
    samples['kfold'] = sample_labels_drugid['kfold'].copy()
    
    return samples, sample_labels

X, y = make_uniform_splits()
data_path = os.path.join('MoA Dataset', 'Samples With Folds')
for fold in X['kfold'].unique():
    joblib.dump(X[X['kfold']==fold], os.path.join(data_path, f'samples_fold_{fold}.pkl'), compress=3)
    joblib.dump(y[X['kfold']==fold], os.path.join(data_path, f'labels_fold_{fold}.pkl'), compress=6)