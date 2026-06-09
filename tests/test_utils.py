import numpy as np
import pytest

from numcompute.utils import set_random_seed, chunk_array, train_test_split, describe_array


def test_set_random_seed_reproducibility():
    set_random_seed(123)
    a = np.random.rand(3)
    set_random_seed(123)
    b = np.random.rand(3)
    assert np.allclose(a, b)


def test_chunk_array_1d():
    arr = np.array([1, 2, 3, 4, 5])
    chunks = chunk_array(arr, 2)

    assert len(chunks) == 3
    assert np.array_equal(chunks[0], np.array([1, 2]))
    assert np.array_equal(chunks[1], np.array([3, 4]))
    assert np.array_equal(chunks[2], np.array([5]))


def test_chunk_array_2d():
    arr = np.array([[1, 2], [3, 4], [5, 6]])
    chunks = chunk_array(arr, 2)

    assert len(chunks) == 2
    assert np.array_equal(chunks[0], np.array([[1, 2], [3, 4]]))
    assert np.array_equal(chunks[1], np.array([[5, 6]]))


def test_chunk_array_invalid_chunk_size():
    arr = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        chunk_array(arr, 0)

    with pytest.raises(ValueError):
        chunk_array(arr, -1)


def test_chunk_array_invalid_dimension():
    arr = np.zeros((2, 2, 2))

    with pytest.raises(ValueError):
        chunk_array(arr, 2)


def test_train_test_split_shapes():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

    assert X_train.shape == (7, 2)
    assert X_test.shape == (3, 2)
    assert y_train.shape == (7,)
    assert y_test.shape == (3,)


def test_train_test_split_shuffle_reproducible():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    result1 = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    result2 = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

    for a, b in zip(result1, result2):
        assert np.array_equal(a, b)


def test_train_test_split_invalid_inputs():
    X = np.arange(10).reshape(5, 2)
    y = np.arange(4)

    with pytest.raises(ValueError):
        train_test_split(X, y)

    with pytest.raises(ValueError):
        train_test_split(np.arange(10).reshape(5, 2), np.arange(5), test_size=1.5)


def test_describe_array():
    arr = np.array([1, 2, 3, np.nan])
    desc = describe_array(arr)

    assert desc["shape"] == (4,)
    assert desc["size"] == 4
    assert desc["nan_count"] == 1
    assert desc["min"] == 1.0
    assert desc["max"] == 3.0
    assert desc["mean"] == pytest.approx(2.0)