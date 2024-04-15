import pandas as pd
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import sys
## https://github.com/ThilinaRajapakse/simpletransformers/blob/1b073ff66c41c06ca531f7916990901dbe977c9d/simpletransformers/classification/classification_model.py#L1375
##  if multi_label:
## logits = logits.sigmoid()

from plot_settings import *

DIR = "PreK"
SEED = 1
nclasses = 7
auprs = []
r80s = []
thr_precisions = []

print("fusion of both")
try:
    df_eval_v  = pd.read_csv('temp_calibrated_video.csv')
    df_eval_a = pd.read_csv('temp_calibrated_text.csv')
except:
    print("for pre fusion you need to run plot_cal_temp2.py file for audio and video")
    sys.exit(-1)

y_test_v = np.array(
    df_eval_v["labels"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" ")).tolist()
)

agg_func = np.multiply #np.min, np.max, np.mean , np.product
##calibration before fusion of modality confidences
#just need to aggefate confidences then

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 7))
for i in range(nclasses):
    # Get true labels and predicted probabilities for one class
    y_true = y_test_v[:, i]
    concat = np.vstack((df_eval_v[f'cal_{i}'].to_numpy(), df_eval_a[f'cal_{i}'].to_numpy())).T
    if agg_func == np.multiply:
        y_prob = np.multiply(concat[:,0], concat[:,1])
    else:
        y_prob = agg_func(concat,axis=1)
    ax = axes[i // 3, i % 3]
    gt = torch.Tensor(y_true)

    # we are just fusing the calibrated probs
    calibrated_probs = y_prob
    plot_multiclass_calibration_curve(y_prob, gt, ax, bins=10)
    ax.set_title(f"Class {i}")

    # find precision at thr
    thr = 0.8
    y_pred_cal = calibrated_probs >= thr
    precision = precision_score(gt, y_pred_cal)
    thr_precisions.append(precision)

    ## caluculate  Recall @ 80% Precision (R@80)
    # tpr, fpr, threshold = roc_curve(y_true, y_prob)
    ## compute threshold where precision is 80%
    ## get the pr curve
    ## find the index of the point where precision is 80%
    aupr_metric = aupr(calibrated_probs, gt)
    r80_metric = r80(calibrated_probs, gt)
    auprs.append(aupr_metric)
    r80s.append(r80_metric)
    print(
        f"AUPR for class {i} is {100*aupr_metric:.2f} and R@80 is {100*r80_metric:.2f} and precision at {thr}: {precision:.2f}"
    )

plt.legend()
plt.tight_layout()
plt.savefig("cal_prefusion.png", dpi=150)
print(f"---- pre fusion (calibration before fusion) for {agg_func.__name__}(T,V) ----")
print(f"average AUPR: {100*np.mean(auprs):.2f}")
print(f"thr precision at {thr}: {100*np.mean(thr_precisions):.2f}")
print(f"average R@80: {100*np.mean(r80s):.2f}")
