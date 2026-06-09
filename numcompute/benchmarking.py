"""Benchmarking utilities for NumCompute-Stream.

This module compares vectorised NumPy implementations against loop-based
implementations and also benchmarks the streaming tree/ensemble pipeline.
Only NumPy and the local package modules are used.
"""

from __future__ import annotations

import time

import numpy as np

from .ensemble import RandomForestClassifier
from .pipeline import Pipeline
from .preprocessing import StandardScaler
from .stream import StreamTrainer, iter_chunks
from .tree import DecisionTreeClassifier


def benchmark_function(func, *args, repeat=5, warmup=1, **kwargs):
    """
    Benchmark a callable repeatedly and return summary statistics.
    """
    if not callable(func):
        raise TypeError("func must be callable.")
    if not isinstance(repeat, int) or repeat < 1:
        raise ValueError("repeat must be a positive integer.")
    if not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer.")

    for _ in range(warmup):
        func(*args, **kwargs)

    times = []
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        times.append(time.perf_counter() - start)

    times = np.asarray(times, dtype=float)
    return {
        "name": getattr(func, "__name__", "callable"),
        "result": result,
        "repeat": repeat,
        "warmup": warmup,
        "mean_time": float(np.mean(times)),
        "median_time": float(np.median(times)),
        "min_time": float(np.min(times)),
        "max_time": float(np.max(times)),
        "std_time": float(np.std(times)),
    }


def compare_functions(func_a, func_b, *args, repeat=5, warmup=1, **kwargs):
    """
    Benchmark two functions and compare their timings.
    """
    a = benchmark_function(func_a, *args, repeat=repeat, warmup=warmup, **kwargs)
    b = benchmark_function(func_b, *args, repeat=repeat, warmup=warmup, **kwargs)

    faster_name = a["name"] if a["mean_time"] <= b["mean_time"] else b["name"]
    slower_time = max(a["mean_time"], b["mean_time"])
    faster_time = min(a["mean_time"], b["mean_time"])
    speedup = slower_time / faster_time if faster_time > 0 else np.inf

    return {
        "first": a,
        "second": b,
        "faster_name": faster_name,
        "speedup_vs_slow": float(speedup),
    }


def format_benchmark_table(results):
    """
    Render a plain-text benchmark summary table.
    """
    if not results:
        return "No benchmark results."

    headers = ["name", "mean_time", "median_time", "min_time", "max_time", "std_time"]
    rows = []
    for r in results:
        rows.append([
            str(r.get("name", "")),
            f"{r.get('mean_time', np.nan):.6e}",
            f"{r.get('median_time', np.nan):.6e}",
            f"{r.get('min_time', np.nan):.6e}",
            f"{r.get('max_time', np.nan):.6e}",
            f"{r.get('std_time', np.nan):.6e}",
        ])

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(row):
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    lines = [fmt(headers), "-+-".join("-" * w for w in widths)]
    lines.extend(fmt(row) for row in rows)
    return "\n".join(lines)


def vectorized_sum_of_squares(x):
    x = np.asarray(x, dtype=float)
    return float(np.sum(x ** 2))


def loop_sum_of_squares(x):
    x = np.asarray(x, dtype=float)
    total = 0.0
    for value in x:
        total += float(value) ** 2
    return float(total)


def vectorized_topk(x, k=10):
    x = np.asarray(x, dtype=float)
    if not isinstance(k, int) or k < 1 or k > x.size:
        raise ValueError("k must satisfy 1 <= k <= len(x).")
    idx = np.argpartition(x, -k)[-k:]
    idx = idx[np.argsort(x[idx])[::-1]]
    return x[idx]


def loop_topk(x, k=10):
    x = np.asarray(x, dtype=float)
    if not isinstance(k, int) or k < 1 or k > x.size:
        raise ValueError("k must satisfy 1 <= k <= len(x).")
    return np.sort(x)[-k:][::-1]


def vectorized_pairwise_distance(X):
    X = np.asarray(X, dtype=float)
    sq = np.sum(X ** 2, axis=1, keepdims=True)
    d2 = sq + sq.T - 2 * X @ X.T
    return np.sqrt(np.maximum(d2, 0.0))


def loop_pairwise_distance(X):
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    out = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            out[i, j] = np.linalg.norm(X[i] - X[j])
    return out


def benchmark_tree_vs_forest(X, y, repeat=3, warmup=1):
    """
    Benchmark a single decision tree against a random forest.
    """
    tree = DecisionTreeClassifier(max_depth=4, random_state=42)
    forest = RandomForestClassifier(n_estimators=7, max_depth=4, random_state=42)

    tree_bench = benchmark_function(tree.fit, X, y, repeat=repeat, warmup=warmup)
    forest_bench = benchmark_function(forest.fit, X, y, repeat=repeat, warmup=warmup)

    return {
        "tree": tree_bench,
        "forest": forest_bench,
        "speedup_vs_slow": max(tree_bench["mean_time"], forest_bench["mean_time"])
        / min(tree_bench["mean_time"], forest_bench["mean_time"]),
    }


def benchmark_streaming_pipeline(X, y, chunk_size=32, repeat=1):
    """
    Benchmark a streaming pipeline with incremental chunk updates.
    """
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=5, max_depth=4, random_state=42)),
    ])
    trainer = StreamTrainer(pipeline=pipe)

    timings = []
    for _ in range(repeat):
        trainer.reset()
        start = time.perf_counter()
        for Xc, yc in iter_chunks(X, y, chunk_size=chunk_size):
            trainer.fit_chunk(Xc, yc)
            trainer.score_chunk(Xc, yc)
        timings.append(time.perf_counter() - start)

    timings = np.asarray(timings, dtype=float)
    return {
        "mean_time": float(np.mean(timings)),
        "median_time": float(np.median(timings)),
        "min_time": float(np.min(timings)),
        "max_time": float(np.max(timings)),
        "std_time": float(np.std(timings)),
        "history": trainer.history_,
        "final_accuracy": float(trainer.result()),
    }


def benchmark_chunk_sizes(X, y, chunk_sizes=(8, 16, 32, 64)):
    """
    Benchmark the streaming pipeline across multiple chunk sizes.
    """
    results = []
    for size in chunk_sizes:
        results.append({
            "chunk_size": int(size),
            "streaming": benchmark_streaming_pipeline(X, y, chunk_size=size, repeat=1),
        })
    return results
