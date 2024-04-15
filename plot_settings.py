import os
import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    roc_auc_score,
    recall_score,
    precision_score,
)
import torch

def get_logits(p):
    # Define a small epsilon value to prevent values of `p` being exactly 0 or 1
    epsilon = 1e-7
    # Clip the probabilities to ensure they are within the range (epsilon, 1 - epsilon)
    p = np.clip(p, epsilon, 1 - epsilon)
    x = np.log(p / (1 - p))
    return torch.as_tensor(x, dtype=torch.float32)

def get_eval_data(run_path):
    fpath = os.path.join(run_path, "eval_df_with_scores.csv")
    df = pd.read_csv(fpath)
    print(df)
    y_test = np.array(
        df["labels"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" ")).tolist()
    )

    fpath_npy = os.path.join(run_path, "simple_transformers_preds.npy")
    test_preds = np.load(fpath_npy)
    assert test_preds.shape[0] == len(
        df
    ), "prediction file and pandas file shape do not match"

    return y_test, df, test_preds

def get_train_data(run_path):
    fpath = os.path.join(run_path,"eval_df_with_scores_train.csv")
    df = pd.read_csv(fpath)
    print(df)
    y_train = np.array(
        df["labels"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" ")).tolist()
    )

    fpath_npy = os.path.join(run_path,  "simple_transformers_preds_train.npy")
    train_preds = np.load(fpath_npy)
    assert train_preds.shape[0] == len(
        df
    ), "prediction file and pandas file shape do not match"

    return y_train, df, train_preds


def compute_metrics(gt, scores):
    """
    Compute metrics based on ground truth values and scores.

    Args:
        gt (array-like): Ground truth values.
        scores (array-like): Scores.

    Returns:
        tuple: A tuple containing the computed metrics.
            - auc (float): The area under the ROC curve.
            - ce (float): The cross-entropy.
            - cestd (float): The cross-entropy standard deviation.

    """
    # auc = roc_auc_score(gt, scores)
    auc = 0
    # sgt = F.logsigmoid(scores * (gt * 2 - 1))
    # ce = -sgt.mean()
    sgt = F.binary_cross_entropy_with_logits(scores, gt.float(), reduction="none")
    # sgt = F.cross_entropy(scores, gt.long(), reduction='none')
    ce = sgt.mean()
    cestd = sgt.std() / len(sgt) ** 0.5
    return auc, float(ce), float(cestd)


## https://github.com/tayden/ood-metrics/blob/main/ood_metrics/metrics.py
def auroc(preds, labels, pos_label=1):
    """Calculate and return the area under the ROC curve using unthresholded predictions on the data and a binary true label.

    preds: array, shape = [n_samples]
           Target normality scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           i.e.: an high value means sample predicted "normal", belonging to the positive class

    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.

    pos_label: label of the positive class (1 by default)
    """
    fpr, tpr, _ = roc_curve(labels, preds, pos_label=pos_label)
    return auc(fpr, tpr)


def aupr(preds, labels, pos_label=1):
    """Calculate and return the area under the Precision Recall curve using unthresholded predictions on the data and a binary true label.

    preds: array, shape = [n_samples]
           Target normality scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           i.e.: an high value means sample predicted "normal", belonging to the positive class

    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.

    pos_label: label of the positive class (1 by default)
    """
    precision, recall, _ = precision_recall_curve(labels, preds, pos_label=pos_label)
    return auc(recall, precision)


def r80(preds, labels, pos_label=1):
    """get recall at at least 80% precision"""
    precision, recall, _ = precision_recall_curve(labels, preds, pos_label=pos_label)
    return recall[np.where(precision >= 0.8)[0][0]]


# https://everyhue.me/posts/multiclass-calibration/
def multiclass_calibration_curve(probs, labels, bins=10):
    """
    Args:
        probs (ndarray):
            NxM predicted probabilities for N examples and M classes.
        labels (ndarray):
            Vector of size N where each entry is an integer class label.
        bins (int):
            Number of bins to divide the prediction probabilities into.
    Returns:
        midpoints (ndarray):
            Midpoint value of each bin
        accuracies (ndarray):
            Fraction of examples that are positive in bin
        mean_confidences:
            Average predicted confidences in each bin
    """
    step_size = 1.0 / bins

    midpoints = []
    mean_confidences = []
    accuracies = []

    for i in range(bins):
        beg = i * step_size
        end = (i + 1) * step_size

        bin_mask = (probs >= beg) & (probs < end)
        bin_cnt = bin_mask.astype(np.float32).sum()
        bin_confs = probs[bin_mask]
        if bin_cnt == 0:
            bin_acc = 0.0
            mean_confidences.append(0.0)
        else:
            bin_acc = labels[bin_mask].sum() / bin_cnt
            mean_confidences.append(np.mean(bin_confs))

        midpoints.append((beg + end) / 2.0)
        accuracies.append(bin_acc)

    return midpoints, accuracies, mean_confidences


def plot_multiclass_calibration_curve(probs, labels, ax, bins=10):
    """
    Plot calibration curve
    """
    midpoints, accuracies, mean_confidences = multiclass_calibration_curve(
        probs, labels, bins=bins
    )
    ax.bar(
        midpoints,
        accuracies,
        width=1.0 / float(bins),
        align="center",
        lw=1,
        ec="#000000",
        fc="#2233aa",
        alpha=1,
        label="Model",
        zorder=0,
    )
    ax.scatter(midpoints, accuracies, lw=2, ec="black", fc="#ffffff", zorder=2)
    ax.plot(
        np.linspace(0, 1.0, 20),
        np.linspace(0, 1.0, 20),
        "--",
        lw=2,
        alpha=0.7,
        color="gray",
        label="Perfectly calibrated",
        zorder=1,
    )
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
