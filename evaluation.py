# -*- coding: utf-8 -*-
import numpy as np
from config import *
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, average_precision_score

def TopKEvaluation(k, pred, label_map):
    y_pred = np.argmax(pred, axis=1)
    sorted_pred = np.argsort(pred, axis=1, kind="mergesort")[:, ::-1] 
    for i in range(len(label_map)):
        for item in sorted_pred[:, :int(k)][i]:
            if item == int(label_map[i:i+1][0]):
                y_pred[i] = item
    
    report = classification_report(label_map, y_pred, target_names=config.target_names, output_dict=True)
    df_report = pd.DataFrame(report).transpose()

    accuracy = accuracy_score(label_map, y_pred)
    
    auc_macro_ovo = accuracy
    auc_macro_ovr = accuracy
    auc_weighted_ovo = accuracy
    auc_weighted_ovr = accuracy
    metrics_dict = {'micro Precision': df_report.loc['accuracy','precision'],
                    'micro Recall': df_report.loc['accuracy','recall'],
                    'micro F1': df_report.loc['accuracy','f1-score'],
                    'macro Precision': df_report.loc['macro avg','precision'],
                    'macro Recall': df_report.loc['macro avg','recall'],
                    'macro F1': df_report.loc['macro avg','f1-score'],
                    'weighted Precision': df_report.loc['weighted avg','precision'],
                    'weighted Recall': df_report.loc['weighted avg','recall'],
                    'weighted F1': df_report.loc['weighted avg','f1-score'],
                    'accuracy': accuracy,
                    'auc macro ovo': auc_macro_ovo,
                    'auc macro ovr': auc_macro_ovr,
                    'auc weighted ovo': auc_weighted_ovo,
                    'auc weighted ovr': auc_weighted_ovr
        }
    return metrics_dict

def Metrics(pred_epochi, target_epochi):
    train_metric_top1 = TopKEvaluation(1, pred_epochi.cpu().detach().numpy(), target_epochi.cpu().detach().numpy())
    metric = {'precision_micro_top1': train_metric_top1['micro Precision'],
                    'recall_micro_top1': train_metric_top1['micro Recall'],
                    'f1_micro_top1': train_metric_top1['micro F1'],
                    'precision_macro_top1': train_metric_top1['macro Precision'],
                    'recall_macro_top1': train_metric_top1['macro Recall'],
                    'f1_macro_top1': train_metric_top1['macro F1'],
                    'precision_weighted_top1': train_metric_top1['weighted Precision'],
                    'recall_weighted_top1': train_metric_top1['weighted Recall'],
                    'f1_weighted_top1': train_metric_top1['weighted F1'],
                    'acc_top1': train_metric_top1['accuracy']
                    }
    return metric
