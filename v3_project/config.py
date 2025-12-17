"""Refactored from V3.ipynb: config.py"""

# NOTE: Auto-refactor (functional split). Review and adjust paths/config as needed.
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

task_names = ['fracture', 'displacement', 'shaft_trans', 'varus', 'articular']
train_losses = []
val_losses = []
train_metrics_history = {task: {'loss': [], 'acc': []} for task in task_names}
val_metrics_history = {task: {'loss': [], 'acc': [], 'auc': []} for task in task_names}
task_name_dict = {
    'fracture':0,
    'displacement':1,
    'shaft_trans':2,
    'varus':3,
    'articular':4,
}

task_names = ['fracture', 'displacement', 'shaft_trans', 'varus', 'articular']
task_titles = [
    'Fracture Classification Accuracy',
    'Displacement Prediction Accuracy',
    'Shaft Translation Accuracy',
    'Varus Malalignment Accuracy',
    'Articular Involvement Accuracy'
]

plt.figure(figsize=(12, 30))

plt.subplot(6, 1, 1)

plt.plot(range(1, len(train_losses)+1), train_losses, 'b--', alpha=0.7, label='Train Loss')

val_x = np.arange(5, 5*len(val_losses)+1, 5)
plt.plot(val_x, val_losses, 'r-', linewidth=2, label='Val Loss')

plt.title('Training & Validation Loss', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss Value', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(fontsize=10)
plt.ylim(0, max(train_losses)*1.1)

for i, task in enumerate(task_names):
    plt.subplot(6, 1, i+2)


    train_acc = train_metrics_history[task]['acc']
    plt.plot(range(1, len(train_acc)+1), train_acc, 'b--', alpha=0.7, label=f'Train {task[:3].upper()} Acc')


    val_acc = val_metrics_history[task]['acc']
    val_x = np.arange(5, 5*len(val_acc)+1, 5)
    plt.plot(val_x, val_acc, 'r-', linewidth=2, label=f'Val {task[:3].upper()} Acc')


    plt.title(task_titles[i], fontsize=12)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Accuracy/AUC', fontsize=10)
    plt.ylim(0.5, 1.05)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='lower right', fontsize=9)

plt.tight_layout(pad=3.0, h_pad=2.0)
plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

name_dict = {
    'fracture':'fracture_classification',
    'displacement':'gt_displacement_greater_equal_to_1cm',
    'shaft_trans':'shaft_translation',
    'varus':'varus_malalignment',
    'articular':'art_involvement',
}

task_meta = {
    'fracture': {
        'type': 'multiclass',
        'labels': train_encoders['fracture_classification'].classes_
    },
    'displacement': {
        'type': 'binary',
        'labels': train_encoders['gt_displacement_greater_equal_to_1cm'].classes_
    },
    'shaft_trans': {
        'type': 'multiclass',
        'labels': train_encoders['shaft_translation'].classes_
    },
    'varus': {
        'type': 'binary',
        'labels': train_encoders['varus_malalignment'].classes_
    },
    'articular': {
        'type': 'multiclass',
        'labels': train_encoders['art_involvement'].classes_
    }
}
