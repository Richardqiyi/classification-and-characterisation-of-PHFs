"""Refactored from V3.ipynb: data_utils.py"""

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

def labels_to_numeric(labels_df):
    label_cols = [
        'fracture_classification',
        'gt_displacement_greater_equal_to_1cm',
        'shaft_translation',
        'varus_malalignment',
        'art_involvement'
    ]
    labels_df = labels_df.loc[:, ['patient_id'] + label_cols]

    encoders = {}
    for col in label_cols:
        le = LabelEncoder().fit(labels_df[col])
        labels_df[col] = le.transform(labels_df[col])
        encoders[col] = le
    return labels_df, encoders

def extract_patient_id(file_path):
    filename = os.path.basename(file_path)
    main_part = filename.split('.')[0]
    desired_part = main_part.split('-')[0]
    return desired_part.lower()

def sort_labels(images, df):

    label_map = {}
    for _, row in df.iterrows():
        pid = row["patient_id"].lower()
        label_map[pid] = {
            'fracture': row['fracture_classification'],
            'displacement': row['gt_displacement_greater_equal_to_1cm'],
            'shaft_trans': row['shaft_translation'],
            'varus': row['varus_malalignment'],
            'articular': row['art_involvement']
        }

    labels = []
    for path in images:
        pid = extract_patient_id(path)
        labels.append([
            label_map[pid]['fracture'],
            label_map[pid]['displacement'],
            label_map[pid]['shaft_trans'],
            label_map[pid]['varus'],
            label_map[pid]['articular']
        ])

    return np.array(labels)
