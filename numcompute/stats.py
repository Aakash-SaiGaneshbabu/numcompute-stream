"""Streaming-descriptive statistics for NumCompute-Stream."""

from __future__ import annotations

from collections import deque

import numpy as np


def _validate_array(arr) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    if x.ndim == 0:
        raise ValueError("arr must have at least one dimension.")
    return x


def _validate_axis(x: np.ndarray, axis):
    if axis is None:
        return None
    if not isinstance(axis, int):
        raise TypeError("axis must be an integer or None.")
    if axis < -x.ndim or axis >= x.ndim:
        raise ValueError(f"axis must be between {-x.ndim} and {x.ndim - 1}.")
    return axis


def mean(arr, axis=None):
    x = _validate_array(arr)
    axis = _validate_axis(x, axis)
    return np.nanmean(x, axis=axis)


def median(arr, axis=None):
    x = _validate_array(arr)
    axis = _validate_axis(x, axis)
    return np.nanmedian(x, axis=axis)


def std(arr, axis=None, ddof=0):
    x = _validate_array(arr)
    axis = _validate_axis(x, axis)
    if not isinstance(ddof, int):
        raise TypeError("ddof must be an integer.")
    return np.nanstd(x, axis=axis, ddof=ddof)


def min(arr, axis=None):  # noqa: A001
    x = _validate_array(arr)
    axis = _validate_axis(x, axis)
    return np.nanmin(x, axis=axis)


def max(arr, axis=None):  # noqa: A001
    x = _validate_array(arr)
    axis = _validate_axis(x, axis)
    return np.nanmax(x, axis=axis)


def histogram(arr, bins=10, range=None):  # noqa: A002
    x = _validate_array(arr)
    if not isinstance(bins, int):
        raise TypeError("bins must be an integer.")
    if bins < 1:
        raise ValueError("bins must be positive.")
    flat = x.ravel()
    flat = flat[~np.isnan(flat)]
    return np.histogram(flat, bins=bins, range=range)


def quantile(arr, q, axis=None, interpolation="linear"):
    x = _validate_array(arr)
    axis = _validate_axis(x, axis)
    if interpolation not in {"linear", "lower", "higher", "midpoint"}:
        raise ValueError("Unsupported interpolation method.")
    q_arr = np.asarray(q, dtype=float)
    scalar = q_arr.ndim == 0
    if np.any((q_arr < 0) | (q_arr > 1)):
        raise ValueError("q must be between 0 and 1.")
    result = np.nanquantile(x, q_arr, axis=axis, method=interpolation)
    if scalar:
        return float(np.asarray(result))
    return result


class StreamingStats:
    """Online descriptive statistics using Welford-style updates."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.n_ = 0
        self.mean_ = 0.0
        self.M2_ = 0.0
        self.min_ = np.inf
        self.max_ = -np.inf
        return self

    def update(self, value):
        value = float(value)
        if np.isnan(value):
            return self
        self.n_ += 1
        delta = value - self.mean_
        self.mean_ += delta / self.n_
        delta2 = value - self.mean_
        self.M2_ += delta * delta2
        if value < self.min_:
            self.min_ = value
        if value > self.max_:
            self.max_ = value
        return self

    def update_many(self, values):
        values = np.asarray(values, dtype=float).ravel()
        for value in values:
            self.update(value)
        return self

    @property
    def count(self):
        return self.n_

    @property
    def mean(self):
        return np.nan if self.n_ == 0 else self.mean_

    @property
    def variance(self):
        return np.nan if self.n_ == 0 else self.M2_ / self.n_

    @property
    def sample_variance(self):
        return np.nan if self.n_ < 2 else self.M2_ / (self.n_ - 1)

    @property
    def std(self):
        return float(np.sqrt(self.variance))

    @property
    def sample_std(self):
        return float(np.sqrt(self.sample_variance))

    @property
    def min(self):  # noqa: A003
        return np.nan if self.n_ == 0 else self.min_

    @property
    def max(self):  # noqa: A003
        return np.nan if self.n_ == 0 else self.max_

    def result(self):
        return self.to_dict()

    def to_dict(self):
        return {
            "count": self.count,
            "mean": self.mean,
            "variance": self.variance,
            "sample_variance": self.sample_variance,
            "std": self.std,
            "sample_std": self.sample_std,
            "min": self.min,
            "max": self.max,
        }


class StreamingHistogram:
    """Chunk-based histogram accumulation."""

    def __init__(self, bins=10, range=None):
        self.bins = bins
        self.range = range
        self.reset()

    def reset(self):
        self._values = deque()
        return self

    def update(self, values):
        values = np.asarray(values, dtype=float).ravel()
        values = values[~np.isnan(values)]
        if values.size:
            self._values.append(values)
        return self

    def result(self):
        if not self._values:
            return np.zeros(self.bins, dtype=int), np.linspace(0, 1, self.bins + 1)
        flat = np.concatenate(list(self._values))
        return np.histogram(flat, bins=self.bins, range=self.range)


class StreamingQuantile:
    """Chunk-based quantile estimation by accumulated observed values."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._values = deque()
        return self

    def update(self, values):
        values = np.asarray(values, dtype=float).ravel()
        values = values[~np.isnan(values)]
        if values.size:
            self._values.append(values)
        return self

    def result(self, q, interpolation="linear"):
        if not self._values:
            return np.nan
        flat = np.concatenate(list(self._values))
        return quantile(flat, q, interpolation=interpolation)


def update_stats(X_chunk, state=None):
    """Convenience API to update or create a StreamingStats state."""
    if state is None:
        state = StreamingStats()
    return state.update_many(X_chunk)