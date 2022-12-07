import numpy as np
import torch
import pandas as pd

# metrics for a model: accuracy, precision, recall, f1 score

def accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    # X_train, X_test, y_train, y_test = np.train_test_split(preds, y, test_size=0.2, random_state=42)
    # clf = LinearSVC(random_state=0, tol=1e-6)
    # clf.fit(X_train, y_train)
    # return clf.score(X_test,y_test)

    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum()/len(correct)
    return acc


def f1_score(preds, y):
    """
    Returns f1 score per batch
    """
    rounded_preds = torch.round(torch.sigmoid(preds))
    tp = (rounded_preds * y).sum()
    tn = ((1 - rounded_preds) * (1 - y)).sum()
    fp = (rounded_preds * (1 - y)).sum()
    fn = ((1 - rounded_preds) * y).sum()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def precision(preds, y):
    """
    Returns precision per batch
    """
    rounded_preds = torch.round(torch.sigmoid(preds))
    tp = (rounded_preds * y).sum()
    fp = (rounded_preds * (1 - y)).sum()
    precision = tp / (tp + fp)
    return precision

def recall(preds, y):
    """
    Returns recall per batch
    """
    rounded_preds = torch.round(torch.sigmoid(preds))
    tp = (rounded_preds * y).sum()
    fn = ((1 - rounded_preds) * y).sum()
    recall = tp / (tp + fn)
    return recall