"""Refactored from V3.ipynb: datasets.py"""

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

train_dicts = [{
    "image": img_path,
    "label": lbl
} for img_path, lbl in zip(train_images, train_labels)]

val_dicts = [{
    "image": img_path,
    "label": lbl
} for img_path, lbl in zip(test_images, test_labels)]

train_transforms = Compose([
    LoadImaged(keys=["image"], image_only=True),
    EnsureChannelFirstd(keys=["image"]),
    ScaleIntensityRanged(keys=["image"], a_min=-500, a_max=1300,b_min=0.0,b_max=1.0,clip=True),
    RandRotated(
    keys=["image"],
    range_x=6.0,  # X
    range_y=6.0,  # Y
    range_z=6.0,  # Z
    prob=0.5,
    padding_mode="border",
    mode="bilinear"
    ),
    Resized(keys=["image"], spatial_size=(128,128,128)),

])

val_transforms = Compose([
    LoadImaged(keys=["image"], image_only=True),
    EnsureChannelFirstd(keys=["image"]),
    ScaleIntensityRanged(keys=["image"], a_min=-500, a_max=1300,b_min=0.0,b_max=1.0,clip=True),

    Resized(keys=["image"], spatial_size=(128,128,128)),

])

train_ds = CacheDataset(
    data=train_dicts,
    transform=train_transforms,
    num_workers=4
)

val_ds = CacheDataset(
    data=val_dicts,
    transform=val_transforms,
    num_workers=4
)

train_loader = DataLoader(
    train_ds,
    batch_size=4,
    sampler=RandomSampler(train_ds),
    num_workers=4,
)

val_loader = DataLoader(
    val_ds,
    batch_size=4,
    sampler=SequentialSampler(val_ds),
    num_workers=4
)
