import numpy as np
import pytest

from numcompute.stats import (
    mean,
    median,
    std,
    min,
    max,
    histogram,
    quantile,
    StreamingStats,
    StreamingHistogram,
    StreamingQuantile,
    update_stats,
)


def test_basic_statistics():
    arr = np.array([1, 2, 3, 4, 5])

    assert mean(arr) == 3
    assert median(arr) == 3
    assert std(arr) == np.std(arr)
    assert min(arr) == 1
    assert max(arr) == 5


def test_histogram():
    arr = np.array([1, 2, 3, 4, 5])
    counts, bins = histogram(arr, bins=5)

    assert len(counts) == 5
    assert len(bins) == 6


def test_quantile():
    arr = np.array([1, 2, 3, 4, 5])

    assert quantile(arr, 0.5) == 3.0
    assert quantile(arr, 0.0) == 1.0
    assert quantile(arr, 1.0) == 5.0


def test_quantile_invalid_q():
    with pytest.raises(ValueError):
        quantile([1, 2, 3], 1.5)


def test_streaming_stats():
    stats = StreamingStats()

    stats.update_many([1, 2, 3, 4, 5])

    assert stats.count == 5
    assert stats.mean == 3.0
    assert stats.min == 1
    assert stats.max == 5


def test_streaming_histogram():
    h = StreamingHistogram(bins=5)

    h.update([1, 2, 3])
    h.update([4, 5])

    counts, bins = h.result()

    assert len(counts) == 5
    assert counts.sum() == 5


def test_streaming_quantile():
    q = StreamingQuantile()

    q.update([1, 2, 3])
    q.update([4, 5])

    assert q.result(0.5) == 3.0


def test_update_stats():
    state = update_stats([1, 2, 3])

    assert state.count == 3
    assert state.mean == 2.0