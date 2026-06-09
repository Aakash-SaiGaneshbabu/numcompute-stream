import numpy as np
import pytest

from numcompute.benchmarking import (
    benchmark_function,
    compare_functions,
    format_benchmark_table,
    vectorized_sum_of_squares,
    loop_sum_of_squares,
    vectorized_topk,
    loop_topk,
    vectorized_pairwise_distance,
    loop_pairwise_distance,
    benchmark_tree_vs_forest,
    benchmark_streaming_pipeline,
    benchmark_chunk_sizes,
)


def test_benchmark_function():
    result = benchmark_function(sum, [1, 2, 3], repeat=2, warmup=1)

    assert result["repeat"] == 2
    assert result["warmup"] == 1
    assert "mean_time" in result
    assert result["result"] == 6


def test_compare_functions():
    result = compare_functions(vectorized_sum_of_squares, loop_sum_of_squares, np.array([1, 2, 3]), repeat=2, warmup=0)

    assert "first" in result
    assert "second" in result
    assert "faster_name" in result
    assert "speedup_vs_slow" in result


def test_format_benchmark_table():
    table = format_benchmark_table([
        {"name": "a", "mean_time": 1.0, "median_time": 1.0, "min_time": 1.0, "max_time": 1.0, "std_time": 0.0},
        {"name": "b", "mean_time": 2.0, "median_time": 2.0, "min_time": 2.0, "max_time": 2.0, "std_time": 0.0},
    ])

    assert "name" in table
    assert "a" in table
    assert "b" in table


def test_vectorized_and_loop_sum_of_squares_match():
    x = np.array([1.0, 2.0, 3.0])

    assert vectorized_sum_of_squares(x) == loop_sum_of_squares(x)


def test_vectorized_and_loop_topk_match():
    x = np.array([5.0, 1.0, 9.0, 3.0, 7.0])

    v = vectorized_topk(x, k=3)
    l = loop_topk(x, k=3)

    assert np.array_equal(v, l)


def test_vectorized_and_loop_pairwise_distance_match():
    X = np.array([[0.0, 0.0], [3.0, 4.0]])

    v = vectorized_pairwise_distance(X)
    l = loop_pairwise_distance(X)

    assert np.allclose(v, l)


def test_benchmark_tree_vs_forest():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    result = benchmark_tree_vs_forest(X, y, repeat=1, warmup=0)

    assert "tree" in result
    assert "forest" in result
    assert "speedup_vs_slow" in result


def test_benchmark_streaming_pipeline():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    result = benchmark_streaming_pipeline(X, y, chunk_size=2, repeat=1)

    assert "mean_time" in result
    assert "history" in result
    assert "final_accuracy" in result


def test_benchmark_chunk_sizes():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    result = benchmark_chunk_sizes(X, y, chunk_sizes=(2, 4))

    assert len(result) == 2
    assert result[0]["chunk_size"] == 2
    assert result[1]["chunk_size"] == 4


def test_benchmark_function_validates_callable():
    with pytest.raises(TypeError):
        benchmark_function(123)