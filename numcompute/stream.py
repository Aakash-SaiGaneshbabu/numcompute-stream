"""Streaming training orchestration for NumCompute-Stream.

This module provides a chunk-wise trainer that can work with either a
standalone model or a full pipeline. It logs per-chunk accuracy, cumulative
accuracy, chunk size, and memory footprint.

Only NumPy is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .metrics import StreamingAccuracy


def _memory_footprint_bytes(*arrays):
    total = 0
    for arr in arrays:
        if arr is not None:
            total += np.asarray(arr).nbytes
    return int(total)


def _as_2d_array(X, name="X"):
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")
    if X.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample.")
    return X


def _as_1d_array(y, name="y"):
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    if y.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample.")
    return y


@dataclass
class StreamTrainer:
    """
    Chunk-wise trainer for streaming learning.

    Parameters
    ----------
    pipeline : object or None, default=None
        A pipeline supporting partial_fit() and predict().
    model : object or None, default=None
        A standalone model supporting partial_fit() and predict().
    metric_tracker : object or None, default=None
        Streaming metric tracker. Defaults to StreamingAccuracy.
    """

    pipeline: object = None
    model: object = None
    metric_tracker: object = None
    history_: list = field(default_factory=list, init=False)
    chunk_index_: int = field(default=0, init=False)

    def __post_init__(self):
        if self.pipeline is None and self.model is None:
            raise ValueError("Either pipeline or model must be provided.")
        if self.metric_tracker is None:
            self.metric_tracker = StreamingAccuracy()

    def reset(self):
        """Reset logging and metric state."""
        self.history_.clear()
        self.chunk_index_ = 0
        if hasattr(self.metric_tracker, "reset"):
            self.metric_tracker.reset()
        return self

    def _predict(self, X):
        if self.pipeline is not None:
            return self.pipeline.predict(X)
        return self.model.predict(X)

    def _partial_fit(self, X, y):
        if self.pipeline is not None:
            if hasattr(self.pipeline, "partial_fit"):
                try:
                    self.pipeline.partial_fit(X, y)
                except TypeError:
                    self.pipeline.partial_fit(X)
                return self
            raise ValueError("Pipeline does not implement partial_fit().")

        if hasattr(self.model, "partial_fit"):
            try:
                self.model.partial_fit(X, y)
            except TypeError:
                self.model.partial_fit(X)
        elif hasattr(self.model, "fit"):
            try:
                self.model.fit(X, y)
            except TypeError:
                self.model.fit(X)
        else:
            raise ValueError("Model must implement fit() or partial_fit().")
        return self

    def fit_chunk(self, X_chunk, y_chunk):
        """
        Update the underlying model or pipeline using one chunk.
        """
        X = _as_2d_array(X_chunk, "X_chunk")
        y = _as_1d_array(y_chunk, "y_chunk")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X_chunk and y_chunk must have the same number of rows.")
        self._partial_fit(X, y)
        return self

    def score_chunk(self, X_chunk, y_chunk):
        """
        Score the current model on one chunk and log the result.
        """
        X = _as_2d_array(X_chunk, "X_chunk")
        y = _as_1d_array(y_chunk, "y_chunk")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X_chunk and y_chunk must have the same number of rows.")

        y_pred = self._predict(X)

        if hasattr(self.metric_tracker, "update"):
            self.metric_tracker.update(y, y_pred)

        chunk_acc = float(np.mean(np.asarray(y_pred) == y))
        self.chunk_index_ += 1

        entry = {
            "chunk_index": self.chunk_index_,
            "chunk_size": int(y.shape[0]),
            "chunk_accuracy": chunk_acc,
            "cumulative_accuracy": float(self.metric_tracker.result()),
            "memory_bytes": _memory_footprint_bytes(X_chunk, y_chunk),
        }
        self.history_.append(entry)
        return entry

    def result(self):
        """Return cumulative metric result."""
        return self.metric_tracker.result()

    def predict(self, X):
        """Predict using the wrapped pipeline or model."""
        return self._predict(X)


def iter_chunks(X, y=None, chunk_size=32, drop_last=False):
    """
    Yield contiguous chunks from X and optional y.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,), optional
    chunk_size : int, default=32
    drop_last : bool, default=False
        If True, drop the final partial chunk.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")

    if y is not None:
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("y must be 1D.")
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y must have matching rows.")

    for start in range(0, X.shape[0], chunk_size):
        stop = min(start + chunk_size, X.shape[0])
        if drop_last and (stop - start) < chunk_size:
            continue
        if y is None:
            yield X[start:stop]
        else:
            yield X[start:stop], y[start:stop]