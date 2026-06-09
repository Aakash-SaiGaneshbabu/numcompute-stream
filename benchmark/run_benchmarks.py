"""Benchmark runner for NumCompute-Stream."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

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


def make_data(seed: int = 42, n_samples: int = 384, n_features: int = 8):
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


def run_pair(label, fn_a, fn_b, *args, repeat=5, warmup=1, **kwargs):
    pair = compare_functions(fn_a, fn_b, *args, repeat=repeat, warmup=warmup, **kwargs)

    first = dict(pair["first"])
    second = dict(pair["second"])
    first["name"] = f"{label} :: {first['name']}"
    second["name"] = f"{label} :: {second['name']}"

    return pair, [first, second]


def print_section(title):
    print("\n" + "=" * len(title))
    print(title)
    print("=" * len(title))


def main():
    X, y = make_data()

    core_rows = []
    comparisons = {}

    pair_specs = [
        ("sum_of_squares", vectorized_sum_of_squares, loop_sum_of_squares, (X.ravel(),), {}),
        ("topk", vectorized_topk, loop_topk, (X.ravel(),), {"k": 10}),
        ("pairwise_distance", vectorized_pairwise_distance, loop_pairwise_distance, (X[:48],), {}),
    ]

    for label, fn_a, fn_b, args, kwargs in pair_specs:
        pair, rows = run_pair(label, fn_a, fn_b, *args, repeat=6, warmup=2, **kwargs)
        comparisons[label] = pair
        core_rows.extend(rows)

    tree_vs_forest = benchmark_tree_vs_forest(X, y, repeat=2, warmup=0)
    streaming_pipeline = benchmark_streaming_pipeline(X, y, chunk_size=32, repeat=2)
    chunk_sizes = benchmark_chunk_sizes(X, y, chunk_sizes=(16, 32, 64))

    print_section("Loop vs Vectorized Benchmarks")
    print(format_benchmark_table(core_rows))

    print_section("Tree vs Forest")
    print(json.dumps(tree_vs_forest, indent=2, default=json_default))

    print_section("Streaming Pipeline")
    print(json.dumps(streaming_pipeline, indent=2, default=json_default))

    print_section("Chunk Size Comparison")
    print(json.dumps(chunk_sizes, indent=2, default=json_default))

    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "benchmark_results.json"

    payload = {
        "core_benchmarks": core_rows,
        "comparisons": comparisons,
        "tree_vs_forest": tree_vs_forest,
        "streaming_pipeline": streaming_pipeline,
        "chunk_sizes": chunk_sizes,
    }

    with out_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=json_default)

    print_section("Saved")
    print(f"Benchmark results saved to: {out_file.resolve()}")


if __name__ == "__main__":
    main()