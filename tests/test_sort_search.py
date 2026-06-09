import numpy as np
import pytest

from numcompute.sort_search import (
    bubble_sort,
    insertion_sort,
    selection_sort,
    merge_sort,
    quick_sort,
    linear_search,
    binary_search,
)


def test_bubble_sort():
    arr = np.array([5, 2, 9, 1, 5, 6])
    result = bubble_sort(arr)
    assert np.array_equal(result, np.array([1, 2, 5, 5, 6, 9]))


def test_insertion_sort():
    arr = np.array([3, 1, 4, 1, 5, 9])
    result = insertion_sort(arr)
    assert np.array_equal(result, np.array([1, 1, 3, 4, 5, 9]))


def test_selection_sort():
    arr = np.array([64, 25, 12, 22, 11])
    result = selection_sort(arr)
    assert np.array_equal(result, np.array([11, 12, 22, 25, 64]))


def test_merge_sort():
    arr = np.array([38, 27, 43, 3, 9, 82, 10])
    result = merge_sort(arr)
    assert np.array_equal(result, np.array([3, 9, 10, 27, 38, 43, 82]))


def test_quick_sort():
    arr = np.array([7, 2, 1, 6, 8, 5, 3, 4])
    result = quick_sort(arr)
    assert np.array_equal(result, np.array([1, 2, 3, 4, 5, 6, 7, 8]))


def test_sort_empty_and_single_element():
    assert np.array_equal(bubble_sort(np.array([])), np.array([]))
    assert np.array_equal(quick_sort(np.array([42])), np.array([42]))


def test_linear_search_found_and_not_found():
    arr = np.array([10, 20, 30, 40, 50])
    assert linear_search(arr, 30) == 2
    assert linear_search(arr, 99) == -1


def test_binary_search_found_and_not_found():
    arr = np.array([10, 20, 30, 40, 50])
    assert binary_search(arr, 40) == 3
    assert binary_search(arr, 99) == -1


def test_search_invalid_dimension():
    arr = np.array([[1, 2], [3, 4]])

    with pytest.raises(ValueError):
        linear_search(arr, 3)

    with pytest.raises(ValueError):
        binary_search(arr, 3)