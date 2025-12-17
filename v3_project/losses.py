"""Refactored from V3.ipynb: losses.py"""

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

class MultiTaskLoss(nn.Module):
    def __init__(self, weights, device="cuda"):
        super().__init__()
        self.weights = weights
        self.loss_fns = nn.ModuleList([

            FocalLoss(
                gamma=2.0,
                use_softmax=True,
                to_onehot_y=True,
                include_background=True
            ).to(device),


            FocalLoss(
                gamma=2.0,
                alpha=0.5,
                use_softmax=False,
            ).to(device),


            FocalLoss(
                gamma=2.0,
                use_softmax=True,
                to_onehot_y=True,
                include_background=True
            ).to(device),


            FocalLoss(
                gamma=2.0,
                alpha=0.5,
                use_softmax=False,
            ).to(device),


            FocalLoss(
                gamma=2.0,
                use_softmax=True,
                to_onehot_y=True,
                include_background=True
            ).to(device),
        ])

    def forward(self, outputs, targets):
        total_loss = 0
        losses = []

        for i, (fn, w) in enumerate(zip(self.loss_fns, self.weights)):
            out = outputs[i]
            tgt = targets[:, i]

            if i in [1,3]:
                out = out.unsqueeze(-1) if out.dim() == 1 else out
                tgt = tgt.float().unsqueeze(-1)
            else:
                tgt = tgt.long()

            loss = w * fn(out, tgt)
            losses.append(loss)
            total_loss += loss

        return total_loss, losses
