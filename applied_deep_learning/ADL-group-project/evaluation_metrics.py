import numpy as np

def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    return TP, TN, FP, FN

def accuracy_score(y_true, y_pred, normalize=True):
    TP, TN, FP, FN = compute_metrics(y_true, y_pred)
    if normalize:
        return (TP + TN)/len(y_true)
    else:
        return TP + TN

def recall_score(y_true, y_pred, pos_label=1, average='binary'):
    if pos_label == 1:
        TP, TN, FP, FN = compute_metrics(y_true, y_pred)
    else:
        TN, TP, FN, FP = compute_metrics(y_true, y_pred)

    if average == 'binary':
        return TP / (TP + FN) if (TP + FN) > 0 else 0
    elif average == 'macro':
        pos = TP / (TP + FN)
        neg = TN / (TN + FP)
        return (pos + neg)/2

def f1_score(y_true, y_pred, pos_label=1, average='binary'):
    if pos_label == 1:
        TP, TN, FP, FN = compute_metrics(y_true, y_pred)
    else:
        TN, TP, FN, FP = compute_metrics(y_true, y_pred)

    if average == 'binary':
        return TP/(TP + 0.5*(FP+FN))
    elif average == 'macro':
        pos = TP/(TP + 0.5*(FP+FN))
        neg = TN/(TN + 0.5*(FN+FP))
        return (pos + neg)/2

def jaccard_score(y_true, y_pred, pos_label=1, average='binary'):
    if pos_label == 1:
        TP, TN, FP, FN = compute_metrics(y_true, y_pred)
    else:
        TN, TP, FN, FP = compute_metrics(y_true, y_pred)

    if average == 'binary':
        return TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
    elif average == 'macro':
        pos = TP / (TP + FP + FN)
        neg = TN / (TN + FN + FP)
        return (pos + neg)/2