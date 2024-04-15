import pandas as pd
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

## https://github.com/ThilinaRajapakse/simpletransformers/blob/1b073ff66c41c06ca531f7916990901dbe977c9d/simpletransformers/classification/classification_model.py#L1375
##  if multi_label:
## logits = logits.sigmoid()

from plot_settings import *

DIR = "PreK"
SEED = 1
RUN_PATH_VIDEO = f"/scratch/sansa/manoj_scratch/runs_videocaption_{DIR}_seed{SEED}"
RUN_PATH_TEXT = f"/scratch/sansa/manoj_scratch/runs_textmodule_{DIR}_seed{SEED}"
nclasses = 7
auprs = []
r80s = []
thr_precisions = []

print("fusion of both")
y_test_v, df_eval_v, preds_test_v = get_eval_data(RUN_PATH_VIDEO)
y_test_a, df_eval_a, preds_test_a = get_eval_data(RUN_PATH_TEXT)

agg_func = np.multiply #np.min, np.max, np.mean , np.product
##calibration after fusion of modality confidences

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 7))
for i in range(nclasses):
    # Get true labels and predicted probabilities for one class
    y_true = y_test_v[:, i]
    concat = np.vstack((df_eval_v[str(i)].to_numpy(), df_eval_a[str(i)].to_numpy())).T
    if agg_func == np.multiply:
        y_prob = np.multiply(concat[:,0], concat[:,1])
    else:
        y_prob = agg_func(concat,axis=1)
    ax = axes[i // 3, i % 3]
    # # rel_diagram_sub(y_true, y_prob, ax, M = 10)
    # plot_multiclass_calibration_curve(y_prob, y_true, ax, bins=10)
    # ax.set_title(f"Class {i}")
    # aupr_metric = aupr(y_prob, y_true)
    # ## caluculate  Recall @ 80% Precision (R@80)
    # # tpr, fpr, threshold = roc_curve(y_true, y_prob)
    # ## compute threshold where precision is 80%
    # ## get the pr curve
    # ## find the index of the point where precision is 80%
    # r80_metric = r80(y_prob, y_true)
    # auprs.append(aupr_metric)
    # r80s.append(r80_metric)
    # print(
    #     f"AUPR for class {i} is {100*aupr_metric:.2f} and R@80 is {100*r80_metric:.2f}"
    # )

    # def _get_logits(X):
    #     X = X + 1e-10
    #     # X /= np.sum(X, axis=-1, keepdims=True)
    #     return torch.as_tensor(np.log(X), dtype=torch.float32)

    uncalibrated_scores = get_logits(y_prob)
    # Temperature-scaling calibration
    gt = torch.Tensor(y_true)

    T = torch.Tensor(1).fill_(0)
    T.requires_grad_()
    temp_optimizer = torch.optim.Adam([T], lr=0.001)
    # TODO: do a early stopping or plateau here
    for iter_idx in range(500):
        temp_optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(
            uncalibrated_scores * torch.exp(-T), gt.float()
        )
        loss.backward()
        temp_optimizer.step()
        # if iter_idx % 100 == 0:
            # print(f"iter {iter_idx} loss {loss:.4f} T {T.item()}")

    print(f" Temperature for class {i}: T {T.item()}")
    ## eval on test set predictions
    gt = torch.Tensor(y_true)
    uncalibrated_scores = get_logits(y_prob)
    calibrated_logits = uncalibrated_scores * torch.exp(-T).detach()

    auc_0, ce, cestd = compute_metrics(gt, uncalibrated_scores)
    print(f"Un-Calibrated class {i} ce: {ce:.4f}, cestd: {cestd:.4f}")
    auc_1, ce, cestd = compute_metrics(gt, calibrated_logits)
    print(f"   Calibrated class {i} ce: {ce:.4f}, cestd: {cestd:.4f}")

    calibrated_probs = torch.sigmoid(calibrated_logits)
    plot_multiclass_calibration_curve(calibrated_probs.numpy(), gt, ax, bins=10)
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
print(f"---- post fusion (calibration after fusion) for {agg_func.__name__}(T,V) ----")
print(f"average AUPR: {100*np.mean(auprs):.2f}")
print(f"thr precision at {thr}: {100*np.mean(thr_precisions):.2f}")
print(f"average R@80: {100*np.mean(r80s):.2f}")
