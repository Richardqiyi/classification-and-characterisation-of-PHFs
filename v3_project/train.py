"""Refactored from V3.ipynb: train.py"""

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

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    task_losses = defaultdict(float)
    task_dict = {
        0:'fracture',
        1:'displacement',
        2:'shaft_trans',
        3:'varus',
        4:'articular',
    }

    preds = {task: [] for task in task_dict.values()}
    truths = {task: [] for task in task_dict.values()}

    for batch in loader:
        inputs = batch["image"].to(device)
        labels = batch["label"].to(device).long()

        optimizer.zero_grad()
        outputs = model(inputs)

        loss, losses = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        for i,loss_val in enumerate(losses):
            task_losses[i] += loss_val.item()

        for i, task_name in task_dict.items():
            if i in [0, 2, 4]:
                pred = outputs[i].argmax(dim=1)
            else:
                out = outputs[i].squeeze(-1) if outputs[i].dim() > 1 else outputs[i]
                pred = (torch.sigmoid(out) > 0.5).long()

            preds[task_name].append(pred.detach().cpu().numpy())
            truths[task_name].append(labels[:, i].cpu().numpy())

    for task in preds:
        preds[task] = np.concatenate(preds[task])
        truths[task] = np.concatenate(truths[task])

    metrics = {}
    for task in task_dict.values():
        metrics[f'{task}_acc'] = accuracy_score(truths[task], preds[task])

    avg_loss = total_loss / len(loader)
    task_avg = {k: v/len(loader) for k, v in task_losses.items()}

    return avg_loss, task_avg, metrics

def validate(model, val_loader, loss_function):
    model.eval()
    total_loss = 0.0
    task_losses = defaultdict(float)
    task_dict = {
        0:'fracture',
        1:'displacement',
        2:'shaft_trans',
        3:'varus',
        4:'articular',
    }

    all_outputs = {task: [] for task in task_dict.values()}
    all_labels = {task: [] for task in task_dict.values()}

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device).long()

            outputs = model(inputs)
            loss, losses = loss_function(outputs, labels)

            total_loss += loss.item()

            for i, task_name in task_dict.items():
                all_outputs[task_name].append(outputs[i].detach().cpu())
                all_labels[task_name].append(labels[:, i].cpu())

    avg_loss = total_loss / len(val_loader)
    detailed_metrics = {}

    cms = []
    for task_name in task_dict.values():
        outputs_tensor = torch.cat(all_outputs[task_name])
        labels_tensor = torch.cat(all_labels[task_name])

        is_multiclass = task_name in ['fracture', 'shaft_trans', 'articular']

        cm, task_metrics = calc_metrics(outputs_tensor, labels_tensor, is_multiclass)
        cms.append(cm)

        task_metrics['loss'] = task_losses[task_name] / len(val_loader)

        detailed_metrics[task_name] = task_metrics

    return cms, avg_loss, detailed_metrics
