"""Visualisation utilities for NumCompute-Stream.

All plots are built with matplotlib and can be displayed inline or saved to file.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .metrics import auc, confusion_matrix, roc_curve


def _as_1d(values, name="values"):
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array-like.")
    return arr


def _finalise(fig, save_path=None, show=True):
    """
    Save/show figure in a consistent way.
    """
    if save_path is not None:
        fig.savefig(str(save_path), bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_metric_over_time(metric_values, title="Metric over time", ylabel="Metric", save_path=None, show=True):
    """
    Plot a metric value across chunks or iterations.
    """
    values = _as_1d(metric_values, "metric_values").astype(float)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, values.size + 1), values, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Chunk")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    return _finalise(fig, save_path=save_path, show=show)


def compare_models(metric1, metric2, labels=("Model 1", "Model 2"), title="Model comparison", ylabel="Metric", save_path=None, show=True):
    """
    Compare two streaming metric histories on the same plot.
    """
    m1 = _as_1d(metric1, "metric1").astype(float)
    m2 = _as_1d(metric2, "metric2").astype(float)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, m1.size + 1), m1, marker="o", label=labels[0])
    ax.plot(np.arange(1, m2.size + 1), m2, marker="s", label=labels[1])
    ax.set_title(title)
    ax.set_xlabel("Chunk")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return _finalise(fig, save_path=save_path, show=show)


def plot_predictions_vs_ground_truth(y_true, y_pred, title="Predictions vs Ground Truth", save_path=None, show=True):
    """
    Plot predicted values against actual values for the latest chunk.
    """
    yt = _as_1d(y_true, "y_true")
    yp = _as_1d(y_pred, "y_pred")

    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(yt, marker="o", label="Ground truth")
    ax.plot(yp, marker="x", label="Prediction")
    ax.set_title(title)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Class / Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return _finalise(fig, save_path=save_path, show=show)


def plot_confusion_matrix(y_true, y_pred, labels=("0", "1"), title="Confusion Matrix", save_path=None, show=True, normalize=False):
    """
    Plot a 2x2 confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred).astype(float)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            text = f"{value:.2f}" if normalize else f"{int(value)}"
            ax.text(j, i, text, ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return _finalise(fig, save_path=save_path, show=show)


def plot_roc_curve(y_true, y_score, title="ROC Curve", save_path=None, show=True):
    """
    Plot a ROC curve and display the AUC in the legend.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    score = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, marker="o", label=f"AUC = {score:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return _finalise(fig, save_path=save_path, show=show)


def plot_streaming_metric(history, metric_name="Accuracy", save_path=None, show=True):
    """
    Convenience wrapper for plotting streaming metric values.
    """
    return plot_metric_over_time(
        history,
        title=f"Streaming {metric_name}",
        ylabel=metric_name,
        save_path=save_path,
        show=show,
    )


def plot_memory_usage(memory_history, save_path=None, show=True):
    """
    Plot memory usage across chunks.
    """
    mem = _as_1d(memory_history, "memory_history").astype(float)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, mem.size + 1), mem, marker="o")
    ax.set_title("Memory Usage Over Time")
    ax.set_xlabel("Chunk")
    ax.set_ylabel("Bytes")
    ax.grid(True, alpha=0.3)

    return _finalise(fig, save_path=save_path, show=show)