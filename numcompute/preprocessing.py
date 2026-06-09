"""Streaming-capable preprocessing utilities for NumCompute-Stream."""

from __future__ import annotations

import numpy as np


def _as_2d_array(X, *, dtype=None, name="X") -> np.ndarray:
    """Validate that X is a non-empty 2D array."""
    arr = np.asarray(X, dtype=dtype)

    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array of shape (n_samples, n_features).")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample.")
    if arr.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one feature.")

    return arr


def _check_feature_count(X: np.ndarray, n_features_in_: int) -> None:
    """Validate that X has the same number of features as the fitted data."""
    if X.shape[1] != n_features_in_:
        raise ValueError(
            f"Input has {X.shape[1]} features, but the transformer was fitted with "
            f"{n_features_in_} features."
        )


def _check_numeric_feature_range(feature_range):
    """Validate and return a numeric feature range tuple."""
    if not isinstance(feature_range, tuple) or len(feature_range) != 2:
        raise ValueError("feature_range must be a length-2 tuple.")

    low, high = feature_range

    if not np.isscalar(low) or not np.isscalar(high):
        raise TypeError("feature_range values must be numeric scalars.")

    low = float(low)
    high = float(high)

    if low >= high:
        raise ValueError("feature_range must satisfy min < max.")

    return low, high


def _nan_column_stats(X: np.ndarray):
    """Return column-wise count, sum, min, max, and valid mask."""
    valid = ~np.isnan(X)
    counts = valid.sum(axis=0).astype(float)
    safe = np.where(valid, X, np.nan)

    col_sum = np.nansum(safe, axis=0)
    col_min = np.where(valid.any(axis=0), np.min(np.where(valid, X, np.inf), axis=0), np.nan)
    col_max = np.where(valid.any(axis=0), np.max(np.where(valid, X, -np.inf), axis=0), np.nan)

    return counts, col_sum, col_min, col_max, valid


class StandardScaler:
    """
    Standardize features using z-score normalisation.

    Supports both batch fit() and incremental partial_fit().
    NaNs are ignored during fitting and preserved during transformation.
    Constant columns are safely handled by using a unit scale.
    """

    def __init__(self):
        self.mean_ = None
        self.var_ = None
        self.std_ = None
        self.n_samples_seen_ = None
        self.n_features_in_ = None

    def fit(self, X):
        """Fit the scaler from scratch."""
        self._reset()
        return self.partial_fit(X)

    def partial_fit(self, X):
        """
        Update running mean/variance from a chunk of data.
        """
        X = _as_2d_array(X, dtype=float)

        counts, col_sum, _, _, valid = _nan_column_stats(X)
        if not np.any(counts > 0):
            return self

        chunk_mean = np.divide(col_sum, counts, out=np.zeros_like(col_sum), where=counts > 0)

        centred = np.where(valid, X - chunk_mean, 0.0)
        chunk_M2 = np.sum(centred ** 2, axis=0)

        if self.mean_ is None:
            self.n_features_in_ = X.shape[1]
            self.n_samples_seen_ = counts.copy()
            self.mean_ = np.where(counts > 0, chunk_mean, 0.0)
            self.var_ = np.divide(chunk_M2, counts, out=np.zeros_like(chunk_M2), where=counts > 0)
        else:
            _check_feature_count(X, self.n_features_in_)

            prev_count = self.n_samples_seen_
            prev_mean = self.mean_
            prev_M2 = self.var_ * prev_count

            total_count = prev_count + counts
            delta = chunk_mean - prev_mean

            new_mean = np.where(
                total_count > 0,
                prev_mean + delta * np.divide(counts, total_count, out=np.zeros_like(counts), where=total_count > 0),
                prev_mean,
            )

            new_M2 = (
                prev_M2
                + chunk_M2
                + (delta ** 2)
                * np.divide(prev_count * counts, total_count, out=np.zeros_like(total_count), where=total_count > 0)
            )

            self.n_samples_seen_ = total_count
            self.mean_ = np.where(total_count > 0, new_mean, prev_mean)
            self.var_ = np.divide(new_M2, total_count, out=np.zeros_like(new_M2), where=total_count > 0)

        self.std_ = np.sqrt(np.where((self.var_ == 0) | np.isnan(self.var_), 1.0, self.var_))
        self.std_ = np.where(np.isnan(self.std_), 1.0, self.std_)

        return self

    def transform(self, X):
        """Scale data using fitted running statistics."""
        if self.mean_ is None or self.std_ is None:
            raise ValueError("StandardScaler must be fitted before transform().")

        X = _as_2d_array(X, dtype=float)
        _check_feature_count(X, self.n_features_in_)

        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def _reset(self):
        self.mean_ = None
        self.var_ = None
        self.std_ = None
        self.n_samples_seen_ = None
        self.n_features_in_ = None


class MinMaxScaler:
    """
    Scale features to a specified numeric range.

    Supports incremental partial_fit() by tracking running minima and maxima.
    """

    def __init__(self, feature_range=(0.0, 1.0)):
        self.feature_range = _check_numeric_feature_range(feature_range)
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None
        self.n_samples_seen_ = None
        self.n_features_in_ = None

    def fit(self, X):
        self._reset()
        return self.partial_fit(X)

    def partial_fit(self, X):
        """Update running minimum and maximum from a chunk."""
        X = _as_2d_array(X, dtype=float)

        counts, _, chunk_min, chunk_max, _ = _nan_column_stats(X)
        if not np.any(counts > 0):
            return self

        if self.data_min_ is None:
            self.data_min_ = chunk_min.copy()
            self.data_max_ = chunk_max.copy()
            self.n_samples_seen_ = counts.copy()
            self.n_features_in_ = X.shape[1]
        else:
            _check_feature_count(X, self.n_features_in_)
            self.data_min_ = np.where(np.isnan(self.data_min_), chunk_min, np.minimum(self.data_min_, chunk_min))
            self.data_max_ = np.where(np.isnan(self.data_max_), chunk_max, np.maximum(self.data_max_, chunk_max))
            self.n_samples_seen_ = self.n_samples_seen_ + counts

        self.data_min_ = np.where(np.isnan(self.data_min_), 0.0, self.data_min_)
        self.data_max_ = np.where(np.isnan(self.data_max_), 0.0, self.data_max_)
        self.data_range_ = self.data_max_ - self.data_min_
        self.data_range_ = np.where(self.data_range_ == 0, 1.0, self.data_range_)

        return self

    def transform(self, X):
        """Scale input data using the fitted min/max statistics."""
        if self.data_min_ is None or self.data_max_ is None:
            raise ValueError("MinMaxScaler must be fitted before transform().")

        X = _as_2d_array(X, dtype=float)
        _check_feature_count(X, self.n_features_in_)

        low, high = self.feature_range
        X_std = (X - self.data_min_) / self.data_range_
        return X_std * (high - low) + low

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def _reset(self):
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None
        self.n_samples_seen_ = None
        self.n_features_in_ = None


class SimpleImputer:
    """
    Replace missing numeric values column-wise.

    Streaming support:
    - mean: updates running sums/counts
    - median: stores observed values and recomputes medians on demand
    - constant: uses a fixed fill value
    """

    def __init__(self, strategy="mean", fill_value=0.0):
        allowed = {"mean", "median", "constant"}
        if strategy not in allowed:
            raise ValueError(f"strategy must be one of {sorted(allowed)}.")

        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = None
        self.n_features_in_ = None

        self._counts = None
        self._sums = None
        self._median_buffers = None

    def fit(self, X):
        self._reset()
        return self.partial_fit(X)

    def partial_fit(self, X):
        """Update the imputation statistics from a chunk."""
        X = _as_2d_array(X, dtype=float)

        counts, col_sum, _, _, valid = _nan_column_stats(X)
        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]

        _check_feature_count(X, self.n_features_in_)

        if self.strategy == "constant":
            self.statistics_ = np.full(self.n_features_in_, float(self.fill_value), dtype=float)
            return self

        if self.strategy == "mean":
            if self._counts is None:
                self._counts = counts.copy()
                self._sums = col_sum.copy()
            else:
                self._counts += counts
                self._sums += col_sum

            self.statistics_ = np.divide(
                self._sums,
                self._counts,
                out=np.full(self.n_features_in_, float(self.fill_value), dtype=float),
                where=self._counts > 0,
            )
            return self

        if self._median_buffers is None:
            self._median_buffers = [[] for _ in range(self.n_features_in_)]

        for j in range(self.n_features_in_):
            col_vals = X[valid[:, j], j]
            if col_vals.size > 0:
                self._median_buffers[j].append(col_vals.copy())

        stats = np.full(self.n_features_in_, float(self.fill_value), dtype=float)
        for j, buf in enumerate(self._median_buffers):
            if buf:
                stats[j] = float(np.median(np.concatenate(buf)))

        self.statistics_ = stats
        return self

    def transform(self, X):
        """Replace NaN values using the learned statistics."""
        if self.statistics_ is None:
            raise ValueError("SimpleImputer must be fitted before transform().")

        X = _as_2d_array(X, dtype=float)
        _check_feature_count(X, self.n_features_in_)

        return np.where(np.isnan(X), self.statistics_, X)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def _reset(self):
        self.statistics_ = None
        self.n_features_in_ = None
        self._counts = None
        self._sums = None
        self._median_buffers = None


class OneHotEncoder:
    """
    One-hot encode categorical columns into a dense integer matrix.

    Streaming support:
    - partial_fit() appends new categories in first-seen order
    - transform() preserves stable ordering of previously learned categories
    """

    def __init__(self, handle_unknown="error", dtype=int):
        if handle_unknown not in {"error", "ignore"}:
            raise ValueError("handle_unknown must be 'error' or 'ignore'.")

        self.handle_unknown = handle_unknown
        self.dtype = dtype
        self.categories_ = None
        self._category_maps = None
        self.feature_indices_ = None
        self.n_features_in_ = None
        self.n_output_features_ = None

    def fit(self, X):
        self._reset()
        return self.partial_fit(X)

    def partial_fit(self, X):
        """
        Update learned categories with a new chunk.
        """
        X = _as_2d_array(X, dtype=object)

        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
            self.categories_ = [[] for _ in range(self.n_features_in_)]
            self._category_maps = [dict() for _ in range(self.n_features_in_)]
        else:
            _check_feature_count(X, self.n_features_in_)

        for j in range(self.n_features_in_):
            seen_new = []
            for value in X[:, j]:
                if value not in self._category_maps[j] and value not in seen_new:
                    seen_new.append(value)

            for value in seen_new:
                self._category_maps[j][value] = len(self.categories_[j])
                self.categories_[j].append(value)

        self._sync_metadata()
        return self

    def transform(self, X):
        """Transform categorical data into one-hot encoded columns."""
        if self.categories_ is None:
            raise ValueError("OneHotEncoder must be fitted before transform().")

        X = _as_2d_array(X, dtype=object)
        _check_feature_count(X, self.n_features_in_)

        n_samples = X.shape[0]
        encoded = np.zeros((n_samples, self.n_output_features_), dtype=self.dtype)
        rows = np.arange(n_samples)

        for j, cats in enumerate(self.categories_):
            cats = np.asarray(cats, dtype=object)
            col = X[:, j]
            matches = col[:, None] == cats[None, :]
            known = np.any(matches, axis=1)

            if self.handle_unknown == "error" and np.any(~known):
                raise ValueError(f"Unknown category found in column {j}.")

            positions = np.argmax(matches, axis=1)
            valid_rows = rows[known]
            valid_cols = self.feature_indices_[j] + positions[known]
            encoded[valid_rows, valid_cols] = 1

        return encoded

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def _sync_metadata(self):
        self.categories_ = [list(cats) for cats in self.categories_]
        sizes = [len(cats) for cats in self.categories_]
        self.feature_indices_ = np.concatenate(([0], np.cumsum(sizes)))
        self.n_output_features_ = int(self.feature_indices_[-1]) if sizes else 0

    def _reset(self):
        self.categories_ = None
        self._category_maps = None
        self.feature_indices_ = None
        self.n_features_in_ = None
        self.n_output_features_ = None