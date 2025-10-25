import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)


def compute_metrics(y_true, y_pred_proba, threshold=0.5):
    y_pred = (np.array(y_pred_proba) >= threshold).astype(int)
    y_true = np.array(y_true)

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_pred_proba)
    }

    return metrics


def compute_confusion_matrix(y_true, y_pred_proba, threshold=0.5):
    y_pred = (np.array(y_pred_proba) >= threshold).astype(int)
    y_true = np.array(y_true)
    return confusion_matrix(y_true, y_pred)


def compute_roc_curve(y_true, y_pred_proba):
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    return fpr, tpr, thresholds


def compute_pr_curve(y_true, y_pred_proba):
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    return precision, recall, thresholds
