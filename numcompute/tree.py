"""Decision tree classifier for NumCompute-Stream.

NumPy-only implementation with:
- depth-limited splits
- Gini / entropy criteria
- streaming-compatible partial_fit()
- predict() and predict_proba()
- max_features support for ensembles


The streaming behaviour is deterministic: each new chunk is appended to the
stored training buffer and the tree is rebuilt from all observed data so far.
This keeps the API online-friendly while staying simple and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

def _as_2d_array(X, *, dtype=float, name="X") -> np.ndarray:
    arr = np.asarray(X, dtype=dtype)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array of shape (n_samples, n_features).")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample.")
    if arr.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one feature.")
    return arr


def _as_1d_array(y, *, name="y") -> np.ndarray:
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample.")
    return arr


def _nan_mask_numeric(X: np.ndarray) -> np.ndarray:
    return ~np.isnan(X).any(axis=1)


def _class_counts(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    counts = np.zeros(classes.size, dtype=float)
    for i, cls in enumerate(classes):
        counts[i] = np.sum(y == cls)
    return counts


def _gini_from_counts(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    return float(1.0 - np.sum(p * p))


def _entropy_from_counts(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


@dataclass
class _TreeNode:
    is_leaf: bool
    prediction: object
    proba: np.ndarray
    impurity: float
    n_samples: int
    class_counts: np.ndarray
    depth: int
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["_TreeNode"] = None
    right: Optional["_TreeNode"] = None


class DecisionTreeClassifier:
    """
    Depth-limited decision tree classifier with streaming-compatible updates.
    """

    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        max_features=None,
        criterion="gini",
        random_state=None,
    ):
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 1):
            raise ValueError("max_depth must be a positive integer or None.")
        if not isinstance(min_samples_split, int) or min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer >= 2.")
        if criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be 'gini' or 'entropy'.")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.random_state = random_state

        self.classes_ = None
        self.n_classes_ = None
        self.n_features_in_ = None
        self.tree_ = None
        self.feature_importances_ = None

        self._X_buffer = None
        self._y_buffer = None
        self._rng = np.random.default_rng(random_state)

    # ---------------------------
    # Public API
    # ---------------------------
    def fit(self, X, y):
        """Fit the classifier on a full dataset."""
        X = _as_2d_array(X, dtype=float)
        y = _as_1d_array(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        mask = _nan_mask_numeric(X)
        X = X[mask]
        y = y[mask]

        if X.shape[0] == 0:
            raise ValueError("No valid rows remain after removing NaNs.")

        self._reset_buffers()
        self._append_buffer(X, y)
        self._fit_from_buffer()
        return self

    def partial_fit(self, X_chunk, y_chunk, classes=None):
        """Incrementally fit on a chunk of data."""
        X = _as_2d_array(X_chunk, dtype=float)
        y = _as_1d_array(y_chunk)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X_chunk and y_chunk must have the same number of rows.")

        mask = _nan_mask_numeric(X)
        X = X[mask]
        y = y[mask]

        if X.shape[0] == 0:
            return self

        if classes is not None:
            classes = np.asarray(classes)
            if classes.ndim != 1:
                raise ValueError("classes must be a 1D array-like.")
            if self.classes_ is None:
                self.classes_ = np.unique(classes)
            else:
                self.classes_ = np.unique(np.concatenate([self.classes_, classes]))

        if self._X_buffer is None:
            self._reset_buffers()

        self._append_buffer(X, y)
        self._fit_from_buffer()
        return self

    def predict(self, X):
        """Predict class labels for rows in X."""
        self._ensure_fitted()
        X = _as_2d_array(X, dtype=float)
        if np.isnan(X).any():
            raise ValueError("X contains NaN values. Please impute missing values before prediction.")

        preds = np.empty(X.shape[0], dtype=self.classes_.dtype)
        for i, row in enumerate(X):
            preds[i] = self._predict_row(row, self.tree_)
        return preds

    def predict_proba(self, X):
        """Predict class membership probabilities for rows in X."""
        self._ensure_fitted()
        X = _as_2d_array(X, dtype=float)
        if np.isnan(X).any():
            raise ValueError("X contains NaN values. Please impute missing values before prediction.")

        proba = np.zeros((X.shape[0], self.n_classes_), dtype=float)
        for i, row in enumerate(X):
            leaf = self._traverse(row, self.tree_)
            proba[i] = leaf.proba
        return proba

    def score(self, X, y):
        """Return classification accuracy on X, y."""
        y_true = _as_1d_array(y)
        y_pred = self.predict(X)
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        return float(np.mean(y_true == y_pred))

    # ---------------------------
    # Internal helpers
    # ---------------------------
    def _ensure_fitted(self):
        if self.tree_ is None or self.classes_ is None:
            raise ValueError("DecisionTreeClassifier must be fitted before use.")

    def _reset_buffers(self):
        self._X_buffer = None
        self._y_buffer = None

    def _append_buffer(self, X, y):
        if self._X_buffer is None:
            self._X_buffer = X.copy()
            self._y_buffer = y.copy()
        else:
            if X.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"Input has {X.shape[1]} features but tree was fitted with {self.n_features_in_} features."
                )
            self._X_buffer = np.vstack([self._X_buffer, X])
            self._y_buffer = np.concatenate([self._y_buffer, y])

        self.n_features_in_ = self._X_buffer.shape[1]

        if self.classes_ is None:
            self.classes_ = np.unique(self._y_buffer)
        else:
            self.classes_ = np.unique(np.concatenate([self.classes_, np.unique(self._y_buffer)]))

        self.n_classes_ = self.classes_.size

    def _fit_from_buffer(self):
        self._rng = np.random.default_rng(self.random_state)
        self.feature_importances_ = np.zeros(self.n_features_in_, dtype=float)

        indices = np.arange(self._X_buffer.shape[0])
        self.tree_ = self._build_tree(indices, depth=0)

        total_importance = self.feature_importances_.sum()
        if total_importance > 0:
            self.feature_importances_ = self.feature_importances_ / total_importance

    def _impurity(self, counts):
        if self.criterion == "gini":
            return _gini_from_counts(counts)
        return _entropy_from_counts(counts)

    def _leaf_node(self, y_subset, depth):
        counts = _class_counts(y_subset, self.classes_)
        total = counts.sum()

        if total == 0:
            prediction = self.classes_[0]
            proba = np.ones(self.n_classes_, dtype=float) / self.n_classes_
            impurity = 0.0
        else:
            prediction = self.classes_[int(np.argmax(counts))]
            proba = counts / total
            impurity = self._impurity(counts)

        return _TreeNode(
            is_leaf=True,
            prediction=prediction,
            proba=proba,
            impurity=impurity,
            n_samples=int(total),
            class_counts=counts,
            depth=depth,
        )

    def _candidate_features(self):
        n_features = self.n_features_in_

        if self.max_features is None:
            return np.arange(n_features)

        if isinstance(self.max_features, int):
            if self.max_features < 1:
                raise ValueError("max_features must be >= 1.")
            k = min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            if not (0 < self.max_features <= 1):
                raise ValueError("float max_features must be in (0, 1].")
            k = max(1, int(np.ceil(self.max_features * n_features)))
        elif isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                k = max(1, int(np.sqrt(n_features)))
            elif self.max_features == "log2":
                k = max(1, int(np.log2(n_features)))
            else:
                raise ValueError("max_features must be None, int, float, 'sqrt', or 'log2'.")
        else:
            raise TypeError("max_features must be None, int, float, or str.")

        return self._rng.choice(n_features, size=k, replace=False)

    def _build_tree(self, indices, depth):
        y_subset = self._y_buffer[indices]
        counts = _class_counts(y_subset, self.classes_)
        n_samples = indices.size
        impurity = self._impurity(counts)

        if (
            n_samples < self.min_samples_split
            or np.max(counts) == n_samples
            or (self.max_depth is not None and depth >= self.max_depth)
            or impurity <= 0.0
        ):
            return self._leaf_node(y_subset, depth)

        best_gain = 0.0
        best_feature = None
        best_threshold = None
        best_left = None
        best_right = None

        candidate_features = self._candidate_features()
        X_subset = self._X_buffer[indices]

        for feature in candidate_features:
            col = X_subset[:, feature]
            unique_vals = np.unique(col)

            if unique_vals.size <= 1:
                continue

            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for threshold in thresholds:
                left_mask = col <= threshold
                left_count = int(np.sum(left_mask))
                right_count = n_samples - left_count

                if left_count == 0 or right_count == 0:
                    continue

                left_indices = indices[left_mask]
                right_indices = indices[~left_mask]

                left_counts = _class_counts(self._y_buffer[left_indices], self.classes_)
                right_counts = _class_counts(self._y_buffer[right_indices], self.classes_)

                left_impurity = self._impurity(left_counts)
                right_impurity = self._impurity(right_counts)

                weighted_child_impurity = (
                    (left_count / n_samples) * left_impurity
                    + (right_count / n_samples) * right_impurity
                )

                gain = impurity - weighted_child_impurity

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = float(threshold)
                    best_left = left_indices
                    best_right = right_indices

        if best_feature is None:
            return self._leaf_node(y_subset, depth)

        self.feature_importances_[best_feature] += best_gain * n_samples

        left_node = self._build_tree(best_left, depth + 1)
        right_node = self._build_tree(best_right, depth + 1)

        return _TreeNode(
            is_leaf=False,
            prediction=self.classes_[int(np.argmax(counts))],
            proba=counts / counts.sum(),
            impurity=impurity,
            n_samples=n_samples,
            class_counts=counts,
            depth=depth,
            feature_index=int(best_feature),
            threshold=float(best_threshold),
            left=left_node,
            right=right_node,
        )

    def _traverse(self, row, node):
        current = node
        while not current.is_leaf:
            if row[current.feature_index] <= current.threshold:
                current = current.left
            else:
                current = current.right
        return current

    def _predict_row(self, row, node):
        return self._traverse(row, node).prediction
