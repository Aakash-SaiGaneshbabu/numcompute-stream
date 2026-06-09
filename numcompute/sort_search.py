"""Sorting and searching algorithms for NumCompute."""

from __future__ import annotations

import numpy as np


def _as_1d_array(a):
    arr = np.asarray(a)
    if arr.ndim != 1:
        raise ValueError("Input must be a 1D array.")
    return arr


def bubble_sort(a):
    arr = np.asarray(a).copy()
    n = arr.size
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr


def insertion_sort(a):
    arr = np.asarray(a).copy()
    for i in range(1, arr.size):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr


def selection_sort(a):
    arr = np.asarray(a).copy()
    n = arr.size
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        if min_idx != i:
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr


def merge_sort(a):
    arr = np.asarray(a).copy()
    n = arr.size
    if n <= 1:
        return arr

    mid = n // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    out = np.empty(n, dtype=arr.dtype)
    i = j = k = 0
    while i < left.size and j < right.size:
        if left[i] <= right[j]:
            out[k] = left[i]
            i += 1
        else:
            out[k] = right[j]
            j += 1
        k += 1

    while i < left.size:
        out[k] = left[i]
        i += 1
        k += 1

    while j < right.size:
        out[k] = right[j]
        j += 1
        k += 1

    return out


def quick_sort(a):
    arr = np.asarray(a).copy()
    if arr.size <= 1:
        return arr

    pivot = arr[arr.size // 2]
    left = arr[arr < pivot]
    middle = arr[arr == pivot]
    right = arr[arr > pivot]
    return np.concatenate([quick_sort(left), middle, quick_sort(right)])


def linear_search(a, target):
    arr = _as_1d_array(a)
    matches = np.where(arr == target)[0]
    return int(matches[0]) if matches.size else -1


def binary_search(a, target):
    arr = np.asarray(a)
    if arr.ndim != 1:
        raise ValueError("Input must be a 1D array.")
    left, right = 0, arr.size - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return int(mid)
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1