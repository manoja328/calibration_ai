import pandas as pd
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from plot_settings import *

DIR = "PreK"
SEED = 1
RUN_PATH_VIDEO = f"/scratch/sansa/manoj_scratch/runs_videocaption_{DIR}_seed{SEED}"
RUN_PATH_TEXT = f"/scratch/sansa/manoj_scratch/runs_textmodule_{DIR}_seed{SEED}"
nclasses = 7
auprs = []
r80s = []
thr_precisions = []

RUN_PATH = RUN_PATH_TEXT
# RUN_PATH = RUN_PATH_VIDEO
y_test, df_eval, preds_test = get_eval_data(RUN_PATH)


fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 7))
for i in range(nclasses):
    # Get true labels and predicted probabilities for one class
    y_true = y_test[:, i]
    y_prob = df_eval[str(i)].to_numpy()
    ax = axes[i // 3, i % 3]
    # rel_diagram_sub(y_true, y_prob, ax, M = 10)
    plot_multiclass_calibration_curve(y_prob, y_true, ax, bins=10)
    ax.set_title(f"Class {i}")
    ## caluculate  Recall @ 80% Precision (R@80)
    # tpr, fpr, threshold = roc_curve(y_true, y_prob)
    ## compute threshold where precision is 80%
    ## get the pr curve
    ## find the index of the point where precision is 80%

    # find precision at thr
    thr = 0.8
    y_pred_cal = y_prob >= thr
    precision = precision_score(y_true, y_pred_cal)
    thr_precisions.append(precision)

    aupr_metric = aupr(y_prob, y_true)
    r80_metric = r80(y_prob, y_true)
    auprs.append(aupr_metric)
    r80s.append(r80_metric)
    print(
        f"AUPR for class {i} is {100*aupr_metric:.2f} and R@80 is {100*r80_metric:.2f} and precision at {thr}: {precision:.2f}"
    )

plt.legend()
plt.tight_layout()
plt.savefig("cal_nocal.png", dpi=150)
print("----- nocal------")
print(RUN_PATH)
print(f"average AUPR: {100*np.mean(auprs):.2f}")
print(f"thr precision at {thr}: {100*np.mean(thr_precisions):.2f}")
print(f"average R@80: {100*np.mean(r80s):.2f}")