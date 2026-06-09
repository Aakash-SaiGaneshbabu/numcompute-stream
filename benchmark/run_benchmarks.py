"""Benchmark runner for NumCompute-Stream."""

from __future__ import annotations

from pathlib import Path
import json
import sys
import time

import numpy as np

# Make sure the project root is importable when running this file directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from numcompute.preprocessing import StandardScaler
from numcompute.tree import DecisionTreeClassifier
from numcompute.ensemble import RandomForestClassifier
from numcompute.pipeline import Pipeline
from numcompute.stream import StreamTrainer, iter_chunks


def benchmark_function(func, *args, repeat=5, warmup=1, **kwargs):
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
    tree = DecisionTreeClassifier(max_depth=4, random_state=42)
    forest = RandomForestClassifier(n_estimators=7, max_depth=4, random_state=42)

    tree_bench = benchmark_function(tree.fit, X, y, repeat=repeat, warmup=warmup)
    forest_bench = benchmark_function(forest.fit, X, y, repeat=repeat, warmup=warmup)

    faster = min(tree_bench["mean_time"], forest_bench["mean_time"])
    slower = max(tree_bench["mean_time"], forest_bench["mean_time"])

    return {
        "tree": tree_bench,
        "forest": forest_bench,
        "speedup_vs_slow": float(slower / faster) if faster > 0 else float("inf"),
    }


def benchmark_streaming_pipeline(X, y, chunk_size=32, repeat=1):
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
    results = []
    for size in chunk_sizes:
        results.append({
            "chunk_size": int(size),
            "streaming": benchmark_streaming_pipeline(X, y, chunk_size=size, repeat=1),
        })
    return results


def make_data(seed=42, n_samples=384, n_features=8):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    noise = 0.35 * rng.normal(size=n_samples)
    score = (
        1.4 * X[:, 0]
        - 0.9 * X[:, 1]
        + 0.6 * X[:, 2]
        + 0.15 * np.sin(np.linspace(0, 6 * np.pi, n_samples))
        + noise
    )
    y = (score > 0).astype(int)
    return X, y


def json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return str(obj)


def main():
    X, y = make_data()

    core_rows = []

    pair_specs = [
        ("sum_of_squares", vectorized_sum_of_squares, loop_sum_of_squares, (X.ravel(),), {}),
        ("topk", vectorized_topk, loop_topk, (X.ravel(),), {"k": 10}),
        ("pairwise_distance", vectorized_pairwise_distance, loop_pairwise_distance, (X[:48],), {}),
    ]

    print("\n=== Loop vs Vectorized Benchmarks ===")
    for label, fn_a, fn_b, args, kwargs in pair_specs:
        pair = compare_functions(fn_a, fn_b, *args, repeat=6, warmup=2, **kwargs)
        first = dict(pair["first"])
        second = dict(pair["second"])
        first["name"] = f"{label} :: {first['name']}"
        second["name"] = f"{label} :: {second['name']}"
        core_rows.extend([first, second])

    print(format_benchmark_table(core_rows))

    tree_vs_forest = benchmark_tree_vs_forest(X, y, repeat=2, warmup=0)
    streaming_pipeline = benchmark_streaming_pipeline(X, y, chunk_size=32, repeat=2)
    chunk_sizes = benchmark_chunk_sizes(X, y, chunk_sizes=(16, 32, 64))

    print("\n=== Tree vs Forest ===")
    print(json.dumps(tree_vs_forest, indent=2, default=json_default))

    print("\n=== Streaming Pipeline ===")
    print(json.dumps(streaming_pipeline, indent=2, default=json_default))

    print("\n=== Chunk Size Comparison ===")
    print(json.dumps(chunk_sizes, indent=2, default=json_default))

    out_dir = Path(__file__).resolve().parent
    out_file = out_dir / "benchmark_results.json"

    payload = {
        "core_benchmarks": core_rows,
        "tree_vs_forest": tree_vs_forest,
        "streaming_pipeline": streaming_pipeline,
        "chunk_sizes": chunk_sizes,
    }

    with out_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=json_default)

    print(f"\nSaved results to: {out_file.resolve()}")


if __name__ == "__main__":
    main()
