#%%
#CUDA_LAUNCH_BLOCKING="0"

'''
Replicates the approach used in the report. A TabNet is used for
classification, without any tricks.
'''
#%% Imports
from tabnet_transfer import TabNetMultilabel, TabNetFreezableMultilabel
from pytorch_tabnet.tab_model import TabNetRegressor
from tqdm import tqdm
import os
import torch
from torch import nn
from dataset import get_datasets
from ray import tune
from copy import deepcopy
import numpy as np
import joblib
from results import ResultsLogger, auc, log_loss
import pandas as pd
from datetime import datetime
from pytorch_tabnet.metrics import Metric
from mixup import TabNetMixup
#%%
dir = os.path.dirname(__file__)
SEED = 23
PATH_TO_RESULTS = os.path.join(dir, 'Models')
DESCRIPTOR = 'baseline-mixup'+'-'+datetime.now().strftime('%F_%H-%M-%S')
model_path = os.path.join(
                'Models',
                DESCRIPTOR
            )
if not os.path.exists(model_path):
    os.makedirs(model_path)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

results_logger = ResultsLogger(
    metrics = [
        ('Mean Log Loss', log_loss)
    ],
    sets = ['test', 'train']
)

# Grid Search params
params = {
    'n_d': 25,
    'n_a': 25,
    'n_steps': tune.grid_search([3, 4]),
    'gamma': 1.3,
    'n_independent': 4,
    'n_shared': 4,
    'momentum': .02,
    'lr': tune.grid_search([2e-2, 2e-1]),
    'weight_decay': 1e-5,
    'n_freeze': 1,
    'fold': 0
}

# Declare variables as global
train = None
test = None
fold = None
model = None
train_results = []
test_loss = []
predictions = []


#%%
def run_training(params):

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
    
    dset, minor_classes = get_datasets(
        params['fold'],
        threshold=100,
        return_classes=True
        )
    major_classes = [
        i 
        for i in dset['major']['train']['y'].columns 
        if i not in minor_classes
        ]
    # Create another dataset, with major and minor sets concatenated
    concat_dset = {
        'train': {},
        'test': {},
        'eval': {}
    }
    for set_ in ['train', 'test', 'eval']:
        for i in ['X', 'y']:
            concat_dset[set_][i] = pd.concat(
                [
                    dset['major'][set_][i],
                    dset['minor'][set_][i]
                ],
                axis = 0
            )

    preds = {'major': {}, 'minor': {}} # Holds predictions. Major/Minor -> Train/Eval

    model = TabNetFreezableMultilabel(
        n_d = params['n_d'],
        n_a = params['n_a'],
        n_steps = params['n_steps'],
        gamma = params['gamma'],
        n_independent = params['n_independent'],
        n_shared = params['n_shared'],
        momentum = params['momentum'],
        lambda_sparse = 0,
        optimizer_fn = torch.optim.Adam,
        optimizer_params = dict(
            lr = params['lr'], 
            weight_decay = params['weight_decay']
        ),
        seed = SEED,
        mask_type = 'entmax',
        scheduler_params = dict(
            mode="min",
            patience=5,
            min_lr=1e-5,
            factor=0.9
        ),
        scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
        verbose=10
    )


    # Fit the model with the major or minor classes
    model.fit(
        concat_dset['train']['X'].values,
        concat_dset['train']['y'].values,
        eval_set = [(concat_dset[c]['X'].values, concat_dset[c]['y'].values) for c in ['train', 'eval']], 
        eval_name = ['train', 'validation'],
        eval_metric = ['logits_ll'],
        loss_fn = torch.nn.functional.binary_cross_entropy_with_logits,
        max_epochs = 200,
        patience = 15,
        batch_size = 64,
        drop_last = False,
        augmentations = TabNetMixup(alpha=.5)
    )
    # Save the model in the intermediate state ('major') or final state, 
    # after fine tuning ('minor')
    model.save_model(os.path.join('Models', DESCRIPTOR+'_report'))

    # Predict for all subsets
    for class_ in dset.keys():
        for set_ in concat_dset.keys():
            preds[class_][set_] = pd.DataFrame(
                model.predict_proba(dset[class_][set_]['X'].values),
                index = dset[class_][set_]['X'].index,
                columns = dset[class_][set_]['y'].columns
            )

    # Send the BCEs to Ray Tune
    tune.report(
        bce_major_train = log_loss(dset['major']['train']['y'][major_classes], preds['major']['train'][major_classes]),
        bce_major_eval = log_loss(dset['major']['eval']['y'][major_classes], preds['major']['eval'][major_classes]),
        bce_major_test = log_loss(dset['major']['test']['y'][major_classes], preds['major']['test'][major_classes]),
        bce_minor_train = log_loss(dset['minor']['train']['y'][minor_classes], preds['minor']['train'][minor_classes]),
        bce_minor_eval = log_loss(dset['minor']['eval']['y'][minor_classes], preds['minor']['eval'][minor_classes]),
        bce_minor_test = log_loss(dset['minor']['test']['y'][minor_classes], preds['minor']['test'][minor_classes]),
        auc_major_train = auc(dset['major']['train']['y'][major_classes], preds['major']['train'][major_classes]),
        auc_major_eval = auc(dset['major']['eval']['y'][major_classes], preds['major']['eval'][major_classes]),
        auc_major_test = auc(dset['major']['test']['y'][major_classes], preds['major']['test'][major_classes]),
        auc_minor_train = auc(dset['minor']['train']['y'][minor_classes], preds['minor']['train'][minor_classes]),
        auc_minor_eval = auc(dset['minor']['eval']['y'][minor_classes], preds['minor']['eval'][minor_classes]),
        auc_minor_test = auc(dset['minor']['test']['y'][minor_classes], preds['minor']['test'][minor_classes])
    )
    # Save predictions
    joblib.dump(
        preds,
        'predictions.pkl',
        compress = 3
    )


# %%
results = None
preds = []
for fold in range(5):

    params['fold'] = fold

    analysis = tune.run(
        run_training, 
        config = params, 
        metric = 'bce_minor_eval', 
        mode = 'min',
        resources_per_trial = {'gpu': 1},
        name = DESCRIPTOR
    )
    
    best_trial = analysis.best_trial
    best_scores = analysis.best_result
    preds.append(
        joblib.load(
            os.path.join(
                analysis.best_logdir,
                'predictions.pkl'
            )
        )
    )
    if results is None:
        results = {}
        for k in best_scores: results[k] = [best_scores[k]]
    else:
        for k in best_scores: results[k].append(best_scores[k])

    joblib.dump(
        analysis, 
        os.path.join(model_path, f'tuning-fold-{fold}.pkl'),
        compress=3
    )

results_keys = list(results.keys())[:12]
for k in results_keys:
    results[k+'_mean'] = np.mean(results[k])
    results[k+'_std'] = np.std(results[k])

print(results)
joblib.dump(
        results, 
        os.path.join(model_path, 'results-dict'+'.pkl'),
        compress=3
    )
joblib.dump(
        preds, 
        os.path.join(model_path, 'predictions'+'.pkl'),
        compress=3
    )

# %%
