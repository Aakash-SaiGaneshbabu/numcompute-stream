import numpy as np
import pytest

from numcompute.rank import rankdata, percentile_rank, top_k


def test_rankdata_average():
    arr = np.array([40, 10, 30, 10])
    result = rankdata(arr, method="average")
    expected = np.array([4.0, 1.5, 3.0, 1.5])
    assert np.allclose(result, expected)


def test_rankdata_min_max_dense_ordinal():
    arr = np.array([40, 10, 30, 10])

    assert np.allclose(rankdata(arr, method="min"), np.array([4.0, 1.0, 3.0, 1.0]))
    assert np.allclose(rankdata(arr, method="max"), np.array([4.0, 2.0, 3.0, 2.0]))
    assert np.allclose(rankdata(arr, method="dense"), np.array([3.0, 1.0, 2.0, 1.0]))
    assert np.allclose(rankdata(arr, method="ordinal"), np.array([4.0, 1.0, 3.0, 2.0]))


def test_rankdata_invalid_method():
    arr = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        rankdata(arr, method="unknown")


def test_rankdata_invalid_dimension():
    arr = np.array([[1, 2], [3, 4]])

    with pytest.raises(ValueError):
        rankdata(arr)


def test_percentile_rank():
    arr = np.array([10, 20, 30, 40])
    assert percentile_rank(arr, 25) == 50.0
    assert percentile_rank(arr, 40) == 100.0


def test_percentile_rank_empty_array():
    arr = np.array([])
    result = percentile_rank(arr, 10)
    assert np.isnan(result)


def test_top_k_largest_and_smallest():
    arr = np.array([5, 1, 9, 3, 7])

    values, idx = top_k(arr, k=2, largest=True)
    assert np.array_equal(values, np.array([9, 7]))
    assert np.array_equal(idx, np.array([2, 4]))

    values, idx = top_k(arr, k=2, largest=False)
    assert np.array_equal(values, np.array([1, 3]))
    assert np.array_equal(idx, np.array([1, 3]))


def test_top_k_invalid_inputs():
    arr = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        top_k(arr, k=0)

    with pytest.raises(ValueError):
        top_k(arr, k=4)

    with pytest.raises(ValueError):
        top_k(np.array([[1, 2], [3, 4]]), k=1)