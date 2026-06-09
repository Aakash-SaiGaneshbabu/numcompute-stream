import numpy as np
import pytest

from numcompute.stats import (
    StreamingHistogram,
    StreamingQuantile,
    StreamingStats,
    histogram,
    max,
    mean,
    median,
    min,
    quantile,
    std,
    update_stats,
)


def test_mean_basic():
    assert mean(np.array([1, 2, 3], dtype=float)) == 2.0


def test_mean_axis():
    arr = np.array([[1, 2, np.nan], [4, 5, 6]], dtype=float)
    out = mean(arr, axis=1)
    assert np.allclose(out, np.array([1.5, 5.0]))


def test_std_matches_numpy():
    arr = np.array([1, 2, 3], dtype=float)
    assert np.isclose(std(arr), np.std(arr))


def test_min_max_ignore_nan():
    arr = np.array([1, np.nan, 3], dtype=float)
    assert min(arr) == 1.0
    assert max(arr) == 3.0


def test_histogram_basic():
    counts, bins = histogram(np.array([1, 2, 3], dtype=float), bins=2)
    assert counts.sum() == 3
    assert bins.shape == (3,)


def test_quantile_basic():
    arr = np.array([1, 2, 3], dtype=float)
    assert quantile(arr, 0.5) == 2.0


def test_streaming_stats():
    stats = StreamingStats().update_many([1, 2, 3, np.nan, 4])
    assert stats.count == 4
    assert np.isclose(stats.mean, 2.5)


def test_update_stats_api():
    stats = update_stats([1, 2, 3])
    assert stats.count == 3
    assert np.isclose(stats.mean, 2.0)


def test_streaming_histogram():
    sh = StreamingHistogram(bins=2).update([1, 2]).update([3, 4])
    counts, _ = sh.result()
    assert counts.sum() == 4


def test_streaming_quantile():
    sq = StreamingQuantile().update([1, 2]).update([3, 4])
    assert np.isclose(sq.result(0.5), 2.5)


def test_invalid_axis_raises():
    with pytest.raises(ValueError):
        mean(np.array([1, 2, 3]), axis=2)
