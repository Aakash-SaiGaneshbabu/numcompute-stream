import numpy as np
import pytest

from numcompute.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    SimpleImputer,
    OneHotEncoder,
)


def test_standard_scaler():
    X = np.array([[1], [2], [3], [4]])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    assert np.isclose(np.mean(X_scaled), 0)
    assert np.isclose(np.std(X_scaled), 1)


def test_standard_scaler_partial_fit():
    scaler = StandardScaler()

    scaler.partial_fit([[1], [2]])
    scaler.partial_fit([[3], [4]])

    assert scaler.mean_[0] == 2.5


def test_minmax_scaler():
    X = np.array([[1], [2], [3]])

    scaler = MinMaxScaler()

    result = scaler.fit_transform(X)

    assert np.min(result) == 0
    assert np.max(result) == 1


def test_simple_imputer_mean():
    X = np.array([
        [1, np.nan],
        [2, 4],
    ])

    imp = SimpleImputer(strategy="mean")

    result = imp.fit_transform(X)

    assert not np.isnan(result).any()


def test_simple_imputer_constant():
    X = np.array([
        [1, np.nan]
    ])

    imp = SimpleImputer(strategy="constant", fill_value=99)

    result = imp.fit_transform(X)

    assert result[0, 1] == 99


def test_one_hot_encoder():
    X = np.array([
        ["red"],
        ["blue"],
        ["red"],
    ])

    enc = OneHotEncoder()

    result = enc.fit_transform(X)

    assert result.shape == (3, 2)


def test_one_hot_encoder_ignore_unknown():
    X = np.array([
        ["red"],
        ["blue"]
    ])

    enc = OneHotEncoder(handle_unknown="ignore")

    enc.fit(X)

    result = enc.transform([["green"]])

    assert result.shape == (1, 2)