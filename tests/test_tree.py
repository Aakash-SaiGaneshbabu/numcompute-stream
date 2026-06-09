import numpy as np
import pytest

from numcompute.tree import DecisionTreeClassifier


def make_toy_data():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    return X, y


def test_tree_fit_predict_score():
    X, y = make_toy_data()

    clf = DecisionTreeClassifier(max_depth=2, min_samples_split=2, random_state=42)
    clf.fit(X, y)

    pred = clf.predict(X)
    proba = clf.predict_proba(X)

    assert np.array_equal(pred, y)
    assert clf.score(X, y) == 1.0
    assert proba.shape == (4, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_tree_partial_fit():
    X1 = np.array([[0.0], [1.0]])
    y1 = np.array([0, 0])
    X2 = np.array([[2.0], [3.0]])
    y2 = np.array([1, 1])

    clf = DecisionTreeClassifier(max_depth=2, min_samples_split=2, random_state=42)
    clf.partial_fit(X1, y1, classes=[0, 1])
    clf.partial_fit(X2, y2)

    pred = clf.predict(np.vstack([X1, X2]))

    assert np.array_equal(pred, np.array([0, 0, 1, 1]))
    assert clf.n_features_in_ == 1
    assert clf.classes_.size == 2


def test_tree_rejects_invalid_params():
    with pytest.raises(ValueError):
        DecisionTreeClassifier(max_depth=0)

    with pytest.raises(ValueError):
        DecisionTreeClassifier(min_samples_split=1)

    with pytest.raises(ValueError):
        DecisionTreeClassifier(criterion="unknown")


def test_tree_rejects_shape_mismatch():
    X, y = make_toy_data()
    clf = DecisionTreeClassifier()

    with pytest.raises(ValueError):
        clf.fit(X, y[:-1])


def test_tree_rejects_nan_in_predict():
    X, y = make_toy_data()
    clf = DecisionTreeClassifier(max_depth=2)
    clf.fit(X, y)

    with pytest.raises(ValueError):
        clf.predict(np.array([[np.nan]]))