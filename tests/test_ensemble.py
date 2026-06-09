import numpy as np
import pytest

from numcompute.ensemble import BaggingClassifier, RandomForestClassifier


def make_toy_data():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    return X, y


def test_bagging_fit_predict_proba():
    X, y = make_toy_data()

    clf = BaggingClassifier(
        n_estimators=3,
        max_depth=2,
        bootstrap=False,
        random_state=42,
    )
    clf.fit(X, y)

    pred = clf.predict(X)
    proba = clf.predict_proba(X)

    assert pred.shape == (4,)
    assert proba.shape == (4, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_bagging_partial_fit():
    X1 = np.array([[0.0], [1.0]])
    y1 = np.array([0, 0])
    X2 = np.array([[2.0], [3.0]])
    y2 = np.array([1, 1])

    clf = BaggingClassifier(
        n_estimators=3,
        max_depth=2,
        bootstrap=False,
        random_state=42,
    )
    clf.partial_fit(X1, y1, classes=[0, 1])
    clf.partial_fit(X2, y2)

    pred = clf.predict(np.vstack([X1, X2]))

    assert pred.shape == (4,)
    assert clf.classes_.size == 2
    assert clf.n_features_in_ == 1


def test_random_forest_repr():
    rf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
    text = repr(rf)

    assert "RandomForestClassifier" in text
    assert "n_estimators=5" in text


def test_ensemble_rejects_invalid_params():
    with pytest.raises(ValueError):
        BaggingClassifier(n_estimators=0)

    with pytest.raises(ValueError):
        BaggingClassifier(n_estimators=-1)