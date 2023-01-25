# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 10:38:22 2021

@author: Marco Teixeira
"""
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from typing import Tuple

class Results:
    def __init__(self, class_names, average = None):
        self.balanced_acc = {x : [] for x in class_names}
        self.acc = {x : [] for x in class_names}
        self.pre = {x : [] for x in class_names}
        self.rec = {x : [] for x in class_names}
        self.f1 = {x : [] for x in class_names}
        self.tp = {x : [] for x in class_names}
        self.tn = {x : [] for x in class_names}
        self.fp = {x : [] for x in class_names}
        self.fn = {x : [] for x in class_names}
        self.spe = {x : [] for x in class_names}
        self.confusion = {x : [] for x in class_names}
        self.average = average

    def specificity(self, tn, fp):
        return tn/(tn+fp)

    def update(self, class_name, y_true, y_predicted):
        if len(np.unique(y_true)) == 2:
            # Binary
            self.balanced_acc[class_name].append(metrics.balanced_accuracy_score(y_true, y_predicted))
            self.acc[class_name].append(metrics.accuracy_score(y_true, y_predicted))
            self.pre[class_name].append(metrics.precision_score(y_true, y_predicted))
            self.rec[class_name].append(metrics.recall_score(y_true, y_predicted))
            self.f1[class_name].append(metrics.f1_score(y_true, y_predicted))
            self.tp[class_name].append(metrics.confusion_matrix(y_true, y_predicted).ravel()[3])
            self.tn[class_name].append(metrics.confusion_matrix(y_true, y_predicted).ravel()[0])
            self.fp[class_name].append(metrics.confusion_matrix(y_true, y_predicted).ravel()[1])
            self.fn[class_name].append(metrics.confusion_matrix(y_true, y_predicted).ravel()[2])
            self.spe[class_name].append(self.specificity(self.tn[class_name][-1], self.fp[class_name][-1]))
            self.confusion[class_name].append(metrics.confusion_matrix(y_true, y_predicted))
        else:
            self.balanced_acc[class_name].append(metrics.balanced_accuracy_score(y_true, y_predicted))
            self.acc[class_name].append(metrics.accuracy_score(y_true, y_predicted))
            self.pre[class_name].append(metrics.precision_score(y_true, y_predicted, average = self.average))
            self.rec[class_name].append(metrics.recall_score(y_true, y_predicted, average = self.average))
            self.f1[class_name].append(metrics.f1_score(y_true, y_predicted, average = self.average))
            self.confusion[class_name].append(metrics.confusion_matrix(y_true, y_predicted))

    def get_means(self):
        '''
        Returns: Balanced Accuracy, Accuracy, Precision, Recall, Specificity, F1, TP, FP, TN, FN.
        '''
        balanced_acc_mean = {}
        acc_mean = {}
        pre_mean = {}
        rec_mean = {}
        spe_mean = {}
        f1_mean = {}
        tp_mean = {}
        fp_mean = {}
        tn_mean = {}
        fn_mean = {}

        for k, v in self.balanced_acc.items():
            balanced_acc_mean[k] = np.mean(v)
        for k, v in self.acc.items():
            acc_mean[k] = np.mean(v)
        for k, v in self.pre.items():
            pre_mean[k] = np.mean(v)
        for k, v in self.rec.items():
            rec_mean[k] = np.mean(v)
        for k, v in self.spe.items():
            spe_mean[k] = np.mean(v)
        for k, v in self.f1.items():
            f1_mean[k] = np.mean(v)
        for k, v in self.tp.items():
            tp_mean[k] = np.mean(v)
        for k, v in self.fp.items():
            fp_mean[k] = np.mean(v)
        for k, v in self.tn.items():
            tn_mean[k] = np.mean(v)
        for k, v in self.fn.items():
            fn_mean[k] = np.mean(v)

        names = ['Balanced Accuracy', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1', 'TP', 'FP', 'TN', 'FN']
        values = [balanced_acc_mean, acc_mean, pre_mean, rec_mean, spe_mean, f1_mean, tp_mean, fp_mean, tn_mean, fn_mean]

        return {k : v for k, v in zip(names, values)}

    def get_stds(self):
        '''
        Returns: Balanced Accuracy, Accuracy, Precision, Recall, Specificity, F1, TP, FP, TN, FN.
        '''
        balanced_acc_mean = {}
        acc_mean = {}
        pre_mean = {}
        rec_mean = {}
        spe_mean = {}
        f1_mean = {}
        tp_mean = {}
        fp_mean = {}
        tn_mean = {}
        fn_mean = {}

        for k, v in self.balanced_acc.items():
            balanced_acc_mean[k] = np.std(v)
        for k, v in self.acc.items():
            acc_mean[k] = np.std(v)
        for k, v in self.pre.items():
            pre_mean[k] = np.std(v)
        for k, v in self.rec.items():
            rec_mean[k] = np.std(v)
        for k, v in self.spe.items():
            spe_mean[k] = np.std(v)
        for k, v in self.f1.items():
            f1_mean[k] = np.std(v)
        for k, v in self.tp.items():
            tp_mean[k] = np.std(v)
        for k, v in self.fp.items():
            fp_mean[k] = np.std(v)
        for k, v in self.tn.items():
            tn_mean[k] = np.std(v)
        for k, v in self.fn.items():
            fn_mean[k] = np.std(v)
        
        names = ['Balanced Accuracy', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1', 'TP', 'FP', 'TN', 'FN']
        values = [balanced_acc_mean, acc_mean, pre_mean, rec_mean, spe_mean, f1_mean, tp_mean, fp_mean, tn_mean, fn_mean]

        return {k : v for k, v in zip(names, values)}
    
class MulticlassResults:
    
    results = {}
    evaluation_metrics = [('Confusion Matrix', metrics.confusion_matrix),
               ('Balanced Accuracy', metrics.balanced_accuracy_score),
               ('Precision', metrics.precision_score),
               ('Recall', metrics.recall_score),
               ('F1', metrics.f1_score),
               ('Accuracy', metrics.accuracy_score)]
    class_names = []
    
    def __init__(self, class_names, metrics = None):
        
        if metrics is not None: self.evaluation_metrics = metrics
        
        self.class_names = class_names
        
        for metric in self.evaluation_metrics:
            self.results[metric[0]] = {c : [] for c in class_names}
            
    def update(self, y_true, y_predicted, class_name):
        
        for metric in self.evaluation_metrics:
            try:
                result_to_append = metric[1](y_true, y_predicted, average = None)
            except TypeError:
                result_to_append = metric[1](y_true, y_predicted)
            
            self.results[metric[0]][class_name].append(result_to_append)
            
    def get_means(self):
        
        means = {metric[0] : {c : [] for c in self.class_names} for metric in self.evaluation_metrics}
        for metric in self.evaluation_metrics:
            for class_ in self.class_names:
                splits_results = self.results[metric[0]][class_]
                means[metric[0]][class_] = np.mean(splits_results, axis = 0)
            
        return means
    
    def get_stds(self):
        
        stds = {metric[0] : {c : [] for c in self.class_names} for metric in self.evaluation_metrics}
        for metric in self.evaluation_metrics:
            for class_ in self.class_names:
                splits_results = self.results[metric[0]][class_]
                stds[metric[0]][class_] = np.std(splits_results, axis = 0)
            
        return stds
    
    def plot_confusion_matrix(self, class_labels):
        
        means = self.get_means().copy()
        
        for class_ in self.class_names:
            n_classes = means['Confusion Matrix'][class_].shape[0]
            with plt.style.context('ggplot'):
                f, ax = plt.subplots(1,1,figsize = (10,10), constrained_layout = True)
                tab = ax.matshow(means['Confusion Matrix'][class_], cmap = 'winter')
                
                try:
                    ax.set_xticks(ticks = np.arange(n_classes))
                    ax.set_xticklabels(class_labels, fontsize = 15)
                    ax.set_yticks(ticks = np.arange(n_classes))
                    ax.set_yticklabels(class_labels, rotation = 90, fontsize = 15, va = 'center')
                except:
                    pass
                for (j, i), value in np.ndenumerate(means['Confusion Matrix'][class_]):
                    ax.text(i, j, int(value), ha='center', va='center', color = 'white', fontsize = 40)
        
                ax.set_xticks(np.arange(n_classes)+1-.5, minor=True)
                ax.set_yticks(np.arange(n_classes)+1-.5, minor=True)
                f.suptitle(f'Confusion Matrix For {class_}', fontsize = 25, color = 'dimgray')
                ax.grid(color = 'w', linewidth = 6, which = 'minor')
                ax.grid(False, which = 'major')
                ax.tick_params(which="minor", bottom=False, left=False, top = False, right = False)
                ax.tick_params(which="major", bottom=False, left=False, top = False, right = False)
                plt.tick_params(axis = 'x', which = 'major', labelbottom = True, bottom=False, labeltop = False)
                colorbar = plt.colorbar(tab, location = 'right', ticks = [0])
                colorbar.ax.set_ylabel('Number of Occurrences', rotation = -90, fontsize = 15)
                colorbar.ax.set_yticklabels([0], fontsize = 15, va = 'bottom')
                colorbar.ax.tick_params(axis = 'y', which="major", left=False, top = False, right = False, labelbottom = True, labelright = True, size = 15)

class ResultsLogger:

    def __init__(
        self,
        metrics,
        classes=None,
        sets = ['train', 'test']
    ):
        if classes is not None:
            self.results = {
                class_: {
                    set_ : {
                        metric[0]: np.array([]) for metric in metrics
                    } for set_ in sets
                } for class_ in classes
            }
            self.means = {
                class_: {
                    set_ : {
                        metric[0]: None for metric in metrics
                    } for set_ in sets
                } for class_ in classes
            }
            self.std = {
                class_: {
                    set_ : {
                        metric[0]: None for metric in metrics
                    } for set_ in sets
                } for class_ in classes
            }
        else:
            self.results = {
                set_ : {
                    metric[0]: np.array([]) for metric in metrics
                } for set_ in sets    
            }
            self.means = {
                set_ : {
                    metric[0]: None for metric in metrics
                } for set_ in sets    
            }
            self.std = {
                set_ : {
                    metric[0]: None for metric in metrics
                } for set_ in sets    
            }

        self.__has_classes = classes is not None
        self.metrics = metrics

    def update(
        self,
        predictions,
        y_true,
        set_,
        class_=None
    ):
        #y_true, predictions = self.sanitize_inputs(y_true, predictions)
        if self.__has_classes:
            for metric_func in self.metrics:
                self.results[class_][set_][metric_func[0]] = np.append(self.results[class_][set_][metric_func[0]], metric_func[1](y_true, predictions))
        else:
            for metric_func in self.metrics:
                self.results[set_][metric_func[0]] = np.append(self.results[set_][metric_func[0]], metric_func[1](y_true, predictions))
    
    def get_means_std(self) -> list:
        if self.__has_classes:
            for class_ in self.results:
                for set_ in self.results[class_]:
                    for metric in self.results[class_][set_]:
                        self.means[class_][set_][metric] = np.mean(self.results[class_][set_][metric])
                        self.std[class_][set_][metric] = np.std(self.results[class_][set_][metric])
        else:
            for set_ in self.results:
                for metric in self.results[set_]:
                    self.means[set_][metric] = np.mean(self.results[set_][metric])
                    self.std[set_][metric] = np.std(self.results[set_][metric])
        return self.means, self.std

    def report(self):
        self.get_means_std()
        if self.__has_classes:
            for class_ in self.results:
                report = {
                    class_: {
                        set_ : {
                            metric: f'{self.means[class_][set_][metric]:.5} +/- {self.std[class_][set_][metric]:.5}' for metric in self.results[class_][set_]
                        } for set_ in self.results[class_]
                    } 
                }
            dframe = pd.DataFrame(report[class_])
            print(f'___ Class {class_} ___')
            print(dframe)
            return None
        else:
            report = {
                set_ : {
                    metric: f'{self.means[set_][metric]:.5} +/- {self.std[set_][metric]:.5}' for metric in self.results[set_]
                } for set_ in self.results
            }
            dframe = pd.DataFrame(report)
            print(dframe) 
            return str(dframe)
    
    def sanitize_inputs(
        self, 
        y: pd.DataFrame, 
        preds: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if np.all(np.any(y, axis=0)):
            return y, preds
        else:
            filtered_preds = preds.iloc[:,np.any(y, axis=0).to_numpy()]
            filtered_y = y.iloc[:,np.any(y, axis=0).to_numpy()]
            return filtered_y, filtered_preds

#%% Sklearn metric functions set to multiclass

def precision(y, preds) -> float:
    return metrics.precision_score(
        y,
        preds,
        average='macro'
    )

def recall(y, preds):
    return metrics.recall_score(
        y,
        preds,
        average='macro'
    )

def auc(y, preds):
    # Only keep columns without all negative labels
    keep = y.columns[y.sum()!=0]

    return metrics.roc_auc_score(
        y[keep],
        preds[keep],
        average='macro',
        multi_class='ovr'
    )

def f1(y, preds):
    return metrics.f1_score(
        y,
        preds,
        average='macro'
    )

def log_loss(y_true, y_pred, normalize=True):
    
    try:
        y_true = y_true.to_numpy()
    except:
        pass
    try:
        y_pred = y_pred.to_numpy()
    except:
        pass
    
    assert y_true.shape == y_pred.shape, 'y_true and y_pred should have the same shape.'
    y_pred = np.clip(y_pred, 1e-5, 1-1e-5)
    
    sample_loss = y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred)
    if np.any(np.isinf(sample_loss)): print(y_pred[np.isnan(sample_loss)])
    
    if normalize:
        return -np.mean(sample_loss)
    else:
        return -np.sum(sample_loss)