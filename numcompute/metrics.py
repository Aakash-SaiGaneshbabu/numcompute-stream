"""Evaluation and streaming metrics for NumCompute-Stream.

This module keeps the batch metrics API and adds streaming trackers
with update(), reset(), and result() methods.
"""

from __future__ import annotations

from collections import deque

import numpy as np

__all__ = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "confusion_matrix",
    "mse",
    "roc_curve",
    "auc",
    "MetricTracker",
    "StreamingAccuracy",
    "StreamingPrecision",
    "StreamingRecall",
    "StreamingF1",
    "StreamingConfusionMatrix",
    "StreamingMSE",
    "StreamingAUC",
]


def _validate_same_shape_1d(y_true, y_pred):
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.ndim != 1 or yp.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D arrays.")
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    return yt, yp


def _drop_nan_pairs(y_true, y_pred):
    if np.issubdtype(y_true.dtype, np.number) and np.issubdtype(y_pred.dtype, np.number):
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        return y_true[mask], y_pred[mask]
    return y_true, y_pred


def _validate_binary(y_true, y_pred):
    y_true, y_pred = _validate_same_shape_1d(y_true, y_pred)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_true, y_pred = _drop_nan_pairs(y_true, y_pred)
    if y_true.size == 0:
        return y_true.astype(int), y_pred.astype(int)
    if not np.all(np.isin(y_true, [0.0, 1.0])):
        raise ValueError("y_true must contain only binary labels 0 and 1.")
    if not np.all(np.isin(y_pred, [0.0, 1.0])):
        raise ValueError("y_pred must contain only binary labels 0 and 1.")
    return y_true.astype(int), y_pred.astype(int)


def _validate_scores(y_true, y_score):
    yt, ys = _validate_same_shape_1d(y_true, y_score)
    yt = np.asarray(yt, dtype=float)
    ys = np.asarray(ys, dtype=float)
    yt, ys = _drop_nan_pairs(yt, ys)
    if yt.size == 0:
        return yt.astype(int), ys.astype(float)
    if not np.all(np.isin(yt, [0.0, 1.0])):
        raise ValueError("y_true must contain only binary labels 0 and 1.")
    return yt.astype(int), ys.astype(float)


def accuracy(y_true, y_pred):
    """Return binary classification accuracy."""
    y_true, y_pred = _validate_binary(y_true, y_pred)
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def precision(y_true, y_pred):
    """Return binary precision."""
    y_true, y_pred = _validate_binary(y_true, y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    denom = tp + fp
    return 0.0 if denom == 0 else float(tp / denom)


def recall(y_true, y_pred):
    """Return binary recall."""
    y_true, y_pred = _validate_binary(y_true, y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    denom = tp + fn
    return 0.0 if denom == 0 else float(tp / denom)


def f1(y_true, y_pred):
    """Return binary F1 score."""
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    denom = p + r
    return 0.0 if denom == 0 else float(2 * p * r / denom)


def confusion_matrix(y_true, y_pred):
    """Return confusion matrix in the standard layout [[TN, FP], [FN, TP]]."""
    y_true, y_pred = _validate_binary(y_true, y_pred)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return np.array([[tn, fp], [fn, tp]], dtype=int)


def mse(y_true, y_pred):
    """Return mean squared error for regression."""
    y_true, y_pred = _validate_same_shape_1d(
        np.asarray(y_true, dtype=float),
        np.asarray(y_pred, dtype=float),
    )
    y_true, y_pred = _drop_nan_pairs(y_true, y_pred)
    if y_true.size == 0:
        return 0.0
    return float(np.mean((y_true - y_pred) ** 2))


def roc_curve(y_true, y_score):
    """
    Compute ROC curve for binary classification.
    """
    y_true, y_score = _validate_scores(y_true, y_score)
    if y_true.size == 0:
        raise ValueError("ROC curve is undefined for empty inputs.")
    desc_idx = np.argsort(-y_score)
    y_true = y_true[desc_idx]
    y_score = y_score[desc_idx]
    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)
    P = np.sum(y_true)
    N = len(y_true) - P
    if P == 0 or N == 0:
        raise ValueError("ROC curve is undefined when only one class is present.")
    tpr = tp / P
    fpr = fp / N
    return fpr, tpr, y_score


def auc(x, y):
    """Compute Area Under Curve using the trapezoidal rule."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1D arrays.")
    if x.shape != y.shape:
        raise ValueError("x and y must have same shape.")
    return float(np.trapezoid(y, x)) if hasattr(np, "trapezoid") else float(np.trapz(y, x))


class MetricTracker:
    """Base class for streaming metric trackers."""

    def __init__(self, window_size=None):
        if window_size is not None:
            if not isinstance(window_size, int) or window_size < 1:
                raise ValueError("window_size must be a positive integer or None.")
        self.window_size = window_size
        self.reset()

    def reset(self):
        self._y_true_chunks = deque()
        self._y_pred_chunks = deque()
        self._n_samples = 0
        return self

    def _append_chunk(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.ndim != 1 or y_pred.ndim != 1:
            raise ValueError("Streaming inputs must be 1D arrays.")
        if y_true.shape != y_pred.shape:
            raise ValueError("Streaming chunks must have the same shape.")
        self._y_true_chunks.append(y_true.copy())
        self._y_pred_chunks.append(y_pred.copy())
        self._n_samples += y_true.size
        if self.window_size is not None:
            self._trim_to_window()

    def _trim_to_window(self):
        if self.window_size is None:
            return
        excess = self._n_samples - self.window_size
        while excess > 0 and self._y_true_chunks:
            first_true = self._y_true_chunks[0]
            first_pred = self._y_pred_chunks[0]
            first_len = first_true.size
            if first_len <= excess:
                self._y_true_chunks.popleft()
                self._y_pred_chunks.popleft()
                self._n_samples -= first_len
                excess -= first_len
            else:
                self._y_true_chunks[0] = first_true[excess:]
                self._y_pred_chunks[0] = first_pred[excess:]
                self._n_samples -= excess
                excess = 0

    def _current_arrays(self):
        if not self._y_true_chunks:
            return np.array([], dtype=float), np.array([], dtype=float)
        return np.concatenate(list(self._y_true_chunks)), np.concatenate(list(self._y_pred_chunks))

    def result(self):
        raise NotImplementedError


class StreamingAccuracy(MetricTracker):
    def update(self, y_true, y_pred):
        y_true, y_pred = _validate_binary(y_true, y_pred)
        self._append_chunk(y_true, y_pred)
        return self

    def result(self):
        y_true, y_pred = self._current_arrays()
        return 0.0 if y_true.size == 0 else accuracy(y_true, y_pred)


class StreamingPrecision(MetricTracker):
    def update(self, y_true, y_pred):
        y_true, y_pred = _validate_binary(y_true, y_pred)
        self._append_chunk(y_true, y_pred)
        return self

    def result(self):
        y_true, y_pred = self._current_arrays()
        return 0.0 if y_true.size == 0 else precision(y_true, y_pred)


class StreamingRecall(MetricTracker):
    def update(self, y_true, y_pred):
        y_true, y_pred = _validate_binary(y_true, y_pred)
        self._append_chunk(y_true, y_pred)
        return self

    def result(self):
        y_true, y_pred = self._current_arrays()
        return 0.0 if y_true.size == 0 else recall(y_true, y_pred)


class StreamingF1(MetricTracker):
    def update(self, y_true, y_pred):
        y_true, y_pred = _validate_binary(y_true, y_pred)
        self._append_chunk(y_true, y_pred)
        return self

    def result(self):
        y_true, y_pred = self._current_arrays()
        return 0.0 if y_true.size == 0 else f1(y_true, y_pred)


class StreamingConfusionMatrix(MetricTracker):
    def update(self, y_true, y_pred):
        y_true, y_pred = _validate_binary(y_true, y_pred)
        self._append_chunk(y_true, y_pred)
        return self

    def result(self):
        y_true, y_pred = self._current_arrays()
        return np.zeros((2, 2), dtype=int) if y_true.size == 0 else confusion_matrix(y_true, y_pred)


class StreamingMSE(MetricTracker):
    def update(self, y_true, y_pred):
        y_true, y_pred = _validate_same_shape_1d(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float))
        y_true, y_pred = _drop_nan_pairs(y_true, y_pred)
        self._append_chunk(y_true, y_pred)
        return self

    def result(self):
        y_true, y_pred = self._current_arrays()
        return 0.0 if y_true.size == 0 else mse(y_true, y_pred)


class StreamingAUC(MetricTracker):
    def update(self, y_true, y_score):
        y_true, y_score = _validate_scores(y_true, y_score)
        self._append_chunk(y_true, y_score)
        return self

    def result(self):
        y_true, y_score = self._current_arrays()
        if y_true.size == 0:
            return np.nan
        try:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            return auc(fpr, tpr)
        except ValueError:
            return np.nan