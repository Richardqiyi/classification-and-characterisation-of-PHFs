"""Refactored from V3.ipynb: plots.py"""

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

plt.rcParams.update({'font.size': 10, 'font.family': 'sans-serif'})


task_config = [
    {'name': 'fracture', 'title': 'ACC', 'color': '#1f77b4'},
    {'name': 'displacement', 'title': 'ACC', 'color': '#ff7f0e'},
    {'name': 'shaft_trans', 'title': 'ACC', 'color': '#2ca02c'},
    {'name': 'varus', 'title': 'ACC', 'color': '#d62728'},
    {'name': 'articular', 'title': 'ACC', 'color': '#9467bd'}
]

plt.figure(figsize=(12, 25))


plt.subplot(6, 1, 1)
train_x = [5*(i+1) for i in range(len(train_losses[4::5]))]
plt.plot(train_x, train_losses[4::5], 'b--', alpha=0.8, linewidth=1.5, label='train loss')


val_x = [5*(i+1) for i in range(len(val_losses))]
plt.plot(val_x, val_losses, 'ro-', markersize=6, linewidth=1.5, label='validation loss')

plt.title('train/val loss', fontsize=14)
plt.xlabel('train (Epoch)', fontsize=12)
plt.ylabel('loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(fontsize=10)
plt.ylim(0, max(train_losses)*1.1)


for i, task in enumerate(task_config):
    ax = plt.subplot(6, 1, i+2)
    color = task['color']


    train_acc = train_metrics_history[task['name']]['acc'][4::5]
    plt.plot(train_x, train_acc, color=color, linestyle='-',
             alpha=0.6, linewidth=1.2, label=f"Train Acc")


    val_acc = val_metrics_history[task['name']]['acc']
    plt.plot(val_x, val_acc, color=color, marker='o',
             markersize=6, linewidth=1.8, label=f"Val Acc")


    best_idx = np.argmax(val_acc)
    best_epoch = val_x[best_idx]
    best_acc = val_acc[best_idx]
    plt.scatter(best_epoch, best_acc, s=120, facecolors='none',
                edgecolors='gold', linewidths=1.5,
                label=f'best: {best_acc:.3f}@{best_epoch}epoch')

    plt.title(task['title'], fontsize=12)
    plt.xlabel('Train epoch (Epoch)', fontsize=10)
    plt.ylabel('Acc', fontsize=10)
    plt.ylim(0.5, 1.05)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='lower right', fontsize=9)

    ax.axhline(y=0.9, color='gray', linestyle=':', alpha=0.4)
    ax.axhline(y=0.8, color='gray', linestyle=':', alpha=0.3)

plt.tight_layout(pad=3.0, h_pad=2.5)
plt.savefig('training_metrics_acc_only.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 30))
for i, (task, meta) in enumerate(task_meta.items(), 1):
    plt.subplot(5, 1, i)

    disp = ConfusionMatrixDisplay(confusion_matrix=cms[i-1],
                                 display_labels=meta['labels'])
    disp.plot(cmap='Blues', ax=plt.gca())
    plt.title(f'{task.capitalize()} Confusion Matrix')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
