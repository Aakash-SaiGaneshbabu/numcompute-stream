"""Tree-based ensemble models for NumCompute-Stream.

This module provides a shared ensemble interface plus streaming-compatible
bagging and random-forest-style classifiers built on DecisionTreeClassifier.
Only NumPy is used.
"""

from __future__ import annotations

import numpy as np

from .tree import DecisionTreeClassifier


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


class EnsembleClassifier:
    """
    Base class for tree-based ensembles.

    Parameters
    ----------
    n_estimators : int, default=10
        Number of trees in the ensemble.
    random_state : int or None, default=None
        Seed for reproducibility.
    """

    def __init__(self, n_estimators=10, random_state=None):
        if not isinstance(n_estimators, int) or n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer.")

        self.n_estimators = n_estimators
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

        self.estimators_ = None
        self.classes_ = None
        self.n_features_in_ = None

    def _reset(self):
        self._rng = np.random.default_rng(self.random_state)
        self.estimators_ = None
        self.classes_ = None
        self.n_features_in_ = None

    def _ensure_estimators(self, tree_kwargs):
        if self.estimators_ is None:
            self.estimators_ = [
                DecisionTreeClassifier(
                    random_state=None if self.random_state is None else self.random_state + i,
                    **tree_kwargs,
                )
                for i in range(self.n_estimators)
            ]

    def _update_classes(self, y):
        classes = np.unique(np.asarray(y))
        if self.classes_ is None:
            self.classes_ = classes
        else:
            self.classes_ = np.unique(np.concatenate([self.classes_, classes]))

    def _bootstrap_indices(self, n_samples):
        return self._rng.integers(0, n_samples, size=n_samples)

    def _align_proba(self, tree, proba):
        aligned = np.zeros((proba.shape[0], self.classes_.size), dtype=float)
        for i, cls in enumerate(tree.classes_):
            pos = np.where(self.classes_ == cls)[0]
            if pos.size:
                aligned[:, pos[0]] = proba[:, i]
        return aligned

    def _majority_vote(self, preds):
        out = np.empty(preds.shape[1], dtype=self.classes_.dtype)
        for j in range(preds.shape[1]):
            values, counts = np.unique(preds[:, j], return_counts=True)
            out[j] = values[np.argmax(counts)]
        return out

    def score(self, X, y):
        y = _as_1d_array(y)
        pred = self.predict(X)
        if pred.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        return float(np.mean(pred == y))


class BaggingClassifier(EnsembleClassifier):
    """
    Streaming-friendly bagging classifier.

    Supports fit(), partial_fit(), predict(), predict_proba(), and score().
    Each chunk update bootstraps the incoming chunk and updates every tree.
    """

    def __init__(
        self,
        n_estimators=10,
        max_depth=5,
        min_samples_split=2,
        max_features=None,
        criterion="gini",
        bootstrap=True,
        random_state=None,
    ):
        super().__init__(n_estimators=n_estimators, random_state=random_state)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.bootstrap = bootstrap

        self._tree_kwargs = dict(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features,
            criterion=criterion,
        )

    def fit(self, X, y):
        X = _as_2d_array(X, dtype=float)
        y = _as_1d_array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        self._reset()
        return self.partial_fit(X, y, classes=np.unique(y))

    def partial_fit(self, X, y, classes=None):
        X = _as_2d_array(X, dtype=float)
        y = _as_1d_array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        if classes is not None:
            classes = np.asarray(classes)
            if classes.ndim != 1:
                raise ValueError("classes must be a 1D array-like.")
            self._update_classes(classes)

        self._update_classes(y)

        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
        elif X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Input has {X.shape[1]} features, but the ensemble was fitted with {self.n_features_in_}."
            )

        self._ensure_estimators(self._tree_kwargs)

        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        for tree in self.estimators_:
            if self.bootstrap:
                idx = self._bootstrap_indices(n_samples)
            else:
                idx = indices

            tree.partial_fit(
                X[idx],
                y[idx],
                classes=np.unique(np.concatenate([self.classes_, np.unique(y[idx])])),
            )

        return self

    def predict_proba(self, X):
        if self.estimators_ is None or self.classes_ is None:
            raise ValueError("BaggingClassifier must be fitted before use.")

        X = _as_2d_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Input has {X.shape[1]} features, but the ensemble was fitted with {self.n_features_in_}."
            )

        proba = np.zeros((X.shape[0], self.classes_.size), dtype=float)
        for tree in self.estimators_:
            proba += self._align_proba(tree, tree.predict_proba(X))
        return proba / self.n_estimators

    def predict(self, X):
        if self.estimators_ is None or self.classes_ is None:
            raise ValueError("BaggingClassifier must be fitted before use.")

        X = _as_2d_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Input has {X.shape[1]} features, but the ensemble was fitted with {self.n_features_in_}."
            )

        preds = np.array([tree.predict(X) for tree in self.estimators_])
        return self._majority_vote(preds)


class RandomForestClassifier(BaggingClassifier):
    """
    Random-forest-style ensemble.

    Uses tree-level feature subsampling via max_features, while keeping the same
    streaming-compatible API as BaggingClassifier.
    """

    def __init__(
        self,
        n_estimators=10,
        max_depth=5,
        min_samples_split=2,
        max_features="sqrt",
        criterion="gini",
        bootstrap=True,
        random_state=None,
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features,
            criterion=criterion,
            bootstrap=bootstrap,
            random_state=random_state,
        )

    def __repr__(self):
        return (
            "RandomForestClassifier("
            f"n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth}, "
            f"min_samples_split={self.min_samples_split}, "
            f"max_features={self.max_features}, "
            f"criterion={self.criterion})"
        )