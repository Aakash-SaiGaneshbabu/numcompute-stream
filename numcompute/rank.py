"""Ranking utilities for NumCompute."""

from __future__ import annotations

import numpy as np


def rankdata(a, method="average"):
    """
    Rank the values in a 1D array.

    Methods:
    - average
    - min
    - max
    - dense
    - ordinal
    """
    x = np.asarray(a)
    if x.ndim != 1:
        raise ValueError("Input must be a 1D array.")

    if method not in {"average", "min", "max", "dense", "ordinal"}:
        raise ValueError("Unsupported ranking method.")

    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=float)

    if method == "ordinal":
        ranks[order] = np.arange(1, x.size + 1)
        return ranks

    sorted_x = x[order]
    unique_vals, start_idx, counts = np.unique(sorted_x, return_index=True, return_counts=True)

    if method == "dense":
        for i, idx in enumerate(start_idx):
            ranks[order[idx: idx + counts[i]]] = i + 1
        return ranks

    for idx, count in zip(start_idx, counts):
        start_rank = idx + 1
        end_rank = idx + count
        if method == "min":
            value = start_rank
        elif method == "max":
            value = end_rank
        else:
            value = (start_rank + end_rank) / 2.0
        ranks[order[idx: idx + count]] = value

    return ranks


def percentile_rank(a, value):
    x = np.asarray(a)
    if x.ndim != 1:
        raise ValueError("Input must be a 1D array.")
    if x.size == 0:
        return np.nan
    return float(np.sum(x <= value) / x.size * 100.0)


def top_k(a, k=1, largest=True):
    x = np.asarray(a)
    if x.ndim != 1:
        raise ValueError("Input must be a 1D array.")
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer.")
    if k > x.size:
        raise ValueError("k cannot exceed the array size.")

    idx = np.argsort(x)
    if largest:
        idx = idx[::-1]
    idx = idx[:k]
    return x[idx], idx