"""Refactored from V3.ipynb: main.py"""

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

def main():
    
    
    # Check environment
    print(torch.__version__)
    print(torch.version.cuda)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    
    train_labels_df = pd.read_csv('train_labels.csv')
    test_labels_df = pd.read_csv('test_labels.csv')
    
    train_labels_df, train_encoders = labels_to_numeric(train_labels_df)
    test_labels_df, test_encoders = labels_to_numeric(test_labels_df)
    # print(train_encoders)
    
    train_images = sorted(glob.glob("new_datasets/one_side_train/*.nii"))
    test_images = sorted(glob.glob("new_datasets/one_side_test/*.nii"))
    
    
    train_labels = sort_labels(train_images, train_labels_df)  # (N,5)
    test_labels = sort_labels(test_images, test_labels_df)
    # print(train_labels[0])
    
    # columns = ["fracture_classification", "gt_displacement_greater_equal_to_1cm",
    #                    "shaft_translation", "varus_malalignment", "art_involvement"]
    # new_df = pd.DataFrame(train_labels, columns=columns)
    # import eda2
    #
    # eda2.visualize_label_distribution(
    #     out_df=new_df,
    #     label_columns=columns,
    #     save_path='tools/add_multi_label_distribution.png'
    # )
    
    criterion = MultiTaskLoss(weights=[1.0, 1.0, 1.0, 1.0, 1.0])
    # optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=10,
    )
    
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=90,
        eta_min=1e-5
    )
    
    start_time = time.time()
    best_val_loss = float('inf')
    
    for epoch in range(100):
    
        train_loss, task_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer
        )
        if epoch<10:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()
    
        train_losses.append(train_loss)
        for task in task_names:
            train_metrics_history[task]['loss'].append(task_loss[task_name_dict[task]])
            train_metrics_history[task]['acc'].append(train_metrics[f'{task}_acc'])
    
    
        if (epoch+1) % 5 == 0:
            _, val_loss, val_metrics = validate(model, val_loader, criterion)
            val_losses.append(val_loss)
    
    
            for task in task_names:
                val_metrics_history[task]['loss'].append(val_metrics[task]['loss'])
                val_metrics_history[task]['acc'].append(val_metrics[task]['accuracy'])
                if task in ['displacement', 'varus']:
                    val_metrics_history[task]['auc'].append(val_metrics[task]['auc'])
    
    
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # torch.save(model.state_dict(), "best_multi_3d_clf4.pth")
    
            print(f"\nEpoch {epoch+1}/100:")
            print(f"lr: {optimizer.param_groups[0]['lr']}")
            print(f"[Train] Loss: {train_loss:.4f}")
            print(f"Fracture acc: {val_metrics['fracture']['accuracy']:.3f}")
    
    
    torch.save(model.state_dict(), "final_multi_3d_clf5.pth")
    print(f"Training time: {(time.time()-start_time)//60} minutes")
    
    cms, val_loss, detailed_metrics = validate(model, val_loader, criterion)
    for task, metrics in detailed_metrics.items():
        print(f"\nTask: {task}")
        print(f"\tOverall Acc: {metrics['accuracy']:.3f}")
        for key in metrics['auc'].keys():
            print(f"\t{train_encoders[name_dict[task]].classes_[key]}\tAUC: {metrics['auc'][key]:.3f} | Sen: {metrics['sensitivity'][key]:.3f} | Spec: {metrics['specificity'][key]:.3f}")


if __name__ == '__main__':
    main()
