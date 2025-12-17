"""Refactored from V3.ipynb: metrics.py"""

import os
import time
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, roc_auc_score
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import SequentialSampler, RandomSampler
from collections import defaultdict
import monai
from monai.losses import FocalLoss
from monai.data import CacheDataset, DataLoader
from monai.transforms import (EnsureChannelFirstd, Compose, Resized, RandGaussianNoised, ScaleIntensityRanged,RandRotated, RandAffined, LoadImaged)
from torch.optim.lr_scheduler import LinearLR,CosineAnnealingLR

def calc_metrics(outputs, labels, is_multiclass):
    metrics = {}

    if is_multiclass:
        probs = torch.softmax(outputs, dim=1)
        preds = probs.argmax(dim=1)
        classes = np.arange(probs.shape[1])
        metrics['accuracy'] = accuracy_score(labels, preds)

        auc_scores = []
        label_bin = label_binarize(labels, classes=np.arange(probs.shape[1]))
        for i in range(probs.shape[1]):
            auc = roc_auc_score(
                label_bin[:,i],
                probs[:,i].numpy(),
                multi_class='ovr')
            auc_scores.append(auc)
        metrics['auc'] = {j: auc_scores[j] for j in range(len(auc_scores))}

        cm = confusion_matrix(labels, preds)
        n_classes = cm.shape[0]
        sensitivity_list = []
        specificity_list = []
        ppv_list = []
        npv_list = []

        for i in range(n_classes):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            fp = np.sum(cm[:, i]) - tp
            tn = np.sum(cm) - tp - fn - fp

            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            ppv_list.append(ppv)
            npv_list.append(npv)
            sensitivity_list.append(sensitivity)
            specificity_list.append(specificity)


        metrics['ppv'] = {j: ppv_list[j] for j in range(len(ppv_list))}
        metrics['npv'] = {j: npv_list[j] for j in range(len(npv_list))}
        metrics['sensitivity'] = {j: sensitivity_list[j] for j in range(len(sensitivity_list))}
        metrics['specificity'] = {j: specificity_list[j] for j in range(len(specificity_list))}

    else:
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).long()
        metrics['accuracy'] = accuracy_score(labels, preds)

        prob_class1 = probs.numpy()
        prob_class0 = 1 - prob_class1
        auc_scores = []

        auc0 = roc_auc_score(labels.numpy(), prob_class0)
        auc1 = roc_auc_score(labels.numpy(), prob_class1)
        auc_scores.append(auc0)
        auc_scores.append(auc1)
        metrics['auc'] = {j: auc_scores[j] for j in range(len(auc_scores))}

        cm = confusion_matrix(labels, preds)
        sensitivity_list = []
        specificity_list = []
        ppv_list = []
        npv_list = []

        for i in range(2):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            fp = np.sum(cm[:, i]) - tp
            tn = np.sum(cm) - tp - fn - fp

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

            ppv_list.append(ppv)
            npv_list.append(npv)
            sensitivity_list.append(sensitivity)
            specificity_list.append(specificity)

        metrics['ppv'] = {j: ppv_list[j] for j in range(len(ppv_list))}
        metrics['npv'] = {j: npv_list[j] for j in range(len(npv_list))}
        metrics['sensitivity'] = {j: sensitivity_list[j] for j in range(len(sensitivity_list))}
        metrics['specificity'] = {j: specificity_list[j] for j in range(len(specificity_list))}

    return cm, metrics
