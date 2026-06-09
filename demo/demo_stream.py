"""Streaming demo for NumCompute-Stream.

This demo shows:
- chunk-based streaming learning
- a single decision tree vs random forest comparison
- per-chunk logging through StreamTrainer
- visualisations saved to disk
- a small benchmark hook
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Make the project root importable when running this file directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from numcompute.preprocessing import StandardScaler
from numcompute.tree import DecisionTreeClassifier
from numcompute.ensemble import RandomForestClassifier
from numcompute.pipeline import Pipeline
from numcompute.stream import StreamTrainer, iter_chunks
from numcompute.visualise import (
    compare_models,
    plot_metric_over_time,
    plot_predictions_vs_ground_truth,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_memory_usage,
)
from numcompute.benchmarking import benchmark_streaming_pipeline


def make_stream_data(seed: int = 7, n_samples: int = 320, n_features: int = 5):
    """Create a simple binary classification stream with mild drift."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))

    drift = np.linspace(-0.6, 0.6, n_samples)
    score = (
        1.4 * X[:, 0]
        - 1.0 * X[:, 1]
        + 0.7 * X[:, 2]
        + 0.15 * np.sin(np.linspace(0, 6 * np.pi, n_samples))
        + drift
        + 0.3 * rng.normal(size=n_samples)
    )
    y = (score > 0).astype(int)
    return X, y


def build_pipeline(model):
    """Wrap a scaler + model in the project pipeline."""
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )


def get_step(pipeline, name: str):
    """Fetch a named step from the pipeline in a safe way."""
    if hasattr(pipeline, "named_steps"):
        return pipeline.named_steps[name]

    if hasattr(pipeline, "steps"):
        for step_name, step in pipeline.steps:
            if step_name == name:
                return step

    raise AttributeError(f"Pipeline does not contain step '{name}'.")


def pipeline_scores(pipeline, X):
    """Return probability-like scores for ROC plotting."""
    scaler = get_step(pipeline, "scale")
    model = get_step(pipeline, "model")

    Xs = scaler.transform(X)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(Xs)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1]
        return np.asarray(proba).ravel()

    return np.asarray(model.predict(Xs), dtype=float)


def run_streaming_experiment(model, X, y, chunk_size=32):
    """Train and evaluate a pipeline one chunk at a time."""
    pipeline = build_pipeline(model)
    trainer = StreamTrainer(pipeline=pipeline)

    chunks = list(iter_chunks(X, y, chunk_size=chunk_size))
    if not chunks:
        raise ValueError("No chunks were produced.")

    history = []

    # First chunk: fit first so the model exists, then score.
    X0, y0 = chunks[0]
    trainer.fit_chunk(X0, y0)
    history.append(trainer.score_chunk(X0, y0))

    # Remaining chunks: score on the current model, then update it.
    for Xc, yc in chunks[1:]:
        history.append(trainer.score_chunk(Xc, yc))
        trainer.fit_chunk(Xc, yc)

    return trainer, history, pipeline


def show_log_preview(logs, name, n=5):
    print(f"\n--- {name} log preview ---")
    for row in logs[:n]:
        print(row)
    if len(logs) > n:
        print("...")
        print(logs[-1])


def history_values(history, key):
    return [float(row[key]) for row in history]


def main():
    X, y = make_stream_data()
    chunk_size = 32

    tree_model = DecisionTreeClassifier(
        max_depth=4,
        min_samples_split=2,
        max_features="sqrt",
        random_state=42,
    )

    forest_model = RandomForestClassifier(
        n_estimators=7,
        max_depth=4,
        min_samples_split=2,
        max_features="sqrt",
        random_state=42,
    )

    tree_trainer, tree_logs, tree_pipeline = run_streaming_experiment(
        tree_model, X, y, chunk_size=chunk_size
    )
    forest_trainer, forest_logs, forest_pipeline = run_streaming_experiment(
        forest_model, X, y, chunk_size=chunk_size
    )

    tree_chunk_acc = history_values(tree_logs, "chunk_accuracy")
    forest_chunk_acc = history_values(forest_logs, "chunk_accuracy")
    tree_cum_acc = history_values(tree_logs, "cumulative_accuracy")
    forest_cum_acc = history_values(forest_logs, "cumulative_accuracy")
    forest_memory = history_values(forest_logs, "memory_bytes")

    last_X, last_y = list(iter_chunks(X, y, chunk_size=chunk_size))[-1]
    last_pred = forest_trainer.predict(last_X)
    forest_scores = pipeline_scores(forest_pipeline, X)
    full_pred = forest_trainer.predict(X)

    print("Streaming demo completed.")
    print(f"Chunks processed: {len(tree_logs)}")
    print(f"Tree final cumulative accuracy: {tree_trainer.result():.4f}")
    print(f"Forest final cumulative accuracy: {forest_trainer.result():.4f}")

    show_log_preview(tree_logs, "Tree")
    show_log_preview(forest_logs, "Forest")

    out_dir = Path(__file__).resolve().parent / "demo_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = compare_models(
        tree_cum_acc,
        forest_cum_acc,
        labels=("Decision Tree", "Random Forest"),
        title="Streaming cumulative accuracy",
        ylabel="Accuracy",
        save_path=out_dir / "compare_cumulative_accuracy.png",
        show=False,
    )
    plt.close(fig)

    fig = plot_metric_over_time(
        tree_chunk_acc,
        title="Decision Tree chunk accuracy",
        ylabel="Accuracy",
        save_path=out_dir / "tree_chunk_accuracy.png",
        show=False,
    )
    plt.close(fig)

    fig = plot_metric_over_time(
        forest_chunk_acc,
        title="Random Forest chunk accuracy",
        ylabel="Accuracy",
        save_path=out_dir / "forest_chunk_accuracy.png",
        show=False,
    )
    plt.close(fig)

    fig = plot_predictions_vs_ground_truth(
        last_y,
        last_pred,
        title="Latest chunk: predictions vs ground truth",
        save_path=out_dir / "latest_chunk_predictions.png",
        show=False,
    )
    plt.close(fig)

    fig = plot_confusion_matrix(
        y,
        full_pred,
        labels=("0", "1"),
        title="Final confusion matrix",
        normalize=False,
        save_path=out_dir / "confusion_matrix.png",
        show=False,
    )
    plt.close(fig)

    fig = plot_roc_curve(
        y,
        forest_scores,
        title="Final ROC curve",
        save_path=out_dir / "roc_curve.png",
        show=False,
    )
    plt.close(fig)

    fig = plot_memory_usage(
        forest_memory,
        save_path=out_dir / "memory_usage.png",
        show=False,
    )
    plt.close(fig)

    summary = {
        "chunk_size": chunk_size,
        "tree_final_cumulative_accuracy": tree_trainer.result(),
        "forest_final_cumulative_accuracy": forest_trainer.result(),
        "tree_history": tree_logs,
        "forest_history": forest_logs,
    }

    with (out_dir / "stream_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Saved plots and summary to: {out_dir.resolve()}")

    # Optional benchmark hook for a quick runtime check.
    bench = benchmark_streaming_pipeline(X, y, chunk_size=chunk_size, repeat=1)
    print(
        "Benchmark summary:",
        json.dumps(
            {
                "mean_time": bench["mean_time"],
                "median_time": bench["median_time"],
                "final_accuracy": bench["final_accuracy"],
            },
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
