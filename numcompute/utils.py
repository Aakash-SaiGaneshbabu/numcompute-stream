"""General utility helpers for NumCompute."""

from __future__ import annotations

import numpy as np


def set_random_seed(seed):
    np.random.seed(seed)


def chunk_array(arr, chunk_size):
    """
    Split a 1D or 2D array into consecutive chunks.
    """
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")
    x = np.asarray(arr)
    if x.ndim not in {1, 2}:
        raise ValueError("arr must be 1D or 2D.")

    chunks = []
    for start in range(0, x.shape[0], chunk_size):
        chunks.append(x[start:start + chunk_size])
    return chunks


def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=None):
    """
    NumPy-only train/test split.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows.")
    if not (0 < test_size < 1):
        raise ValueError("test_size must be between 0 and 1.")

    n = X.shape[0]
    idx = np.arange(n)
    rng = np.random.default_rng(random_state)
    if shuffle:
        rng.shuffle(idx)

    split = int(round(n * (1 - test_size)))
    train_idx = idx[:split]
    test_idx = idx[split:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def describe_array(arr):
    """
    Return a compact numeric summary of an array.
    """
    x = np.asarray(arr, dtype=float)
    return {
        "shape": x.shape,
        "dtype": str(x.dtype),
        "size": int(x.size),
        "nan_count": int(np.isnan(x).sum()) if np.issubdtype(x.dtype, np.number) else 0,
        "min": float(np.nanmin(x)) if x.size else np.nan,
        "max": float(np.nanmax(x)) if x.size else np.nan,
        "mean": float(np.nanmean(x)) if x.size else np.nan,
    }