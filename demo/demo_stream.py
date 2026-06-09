"""Streaming demo for NumCompute-Stream."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from numcompute import (
    StandardScaler,
    DecisionTreeClassifier,
    RandomForestClassifier,
    Pipeline,
    StreamTrainer,
    iter_chunks,
)
from numcompute.visualise import (
    compare_models,
    plot_metric_over_time,
    plot_predictions_vs_ground_truth,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_memory_usage,
)


def make_stream_data(seed: int = 7, n_samples: int = 320, n_features: int = 5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))

    drift = np.linspace(-0.6, 0.6, n_samples)
    signal = (
        1.4 * X[:, 0]
        - 1.0 * X[:, 1]
        + 0.7 * X[:, 2]
        + 0.15 * np.sin(np.linspace(0, 6 * np.pi, n_samples))
        + drift
        + 0.3 * rng.normal(size=n_samples)
    )
    y = (signal > 0).astype(int)
    return X, y


def build_pipeline(model):
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )


def json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return str(obj)


def history_values(history, key):
    return [float(row[key]) for row in history]


def run_streaming_experiment(model, X, y, chunk_size=32):
    pipeline = build_pipeline(model)
    trainer = StreamTrainer(pipeline=pipeline)

    chunks = list(iter_chunks(X, y, chunk_size=chunk_size))
    if not chunks:
        raise ValueError("No chunks were produced.")

    X0, y0 = chunks[0]
    trainer.fit_chunk(X0, y0)
    trainer.score_chunk(X0, y0)

    for Xc, yc in chunks[1:]:
        trainer.score_chunk(Xc, yc)
        trainer.fit_chunk(Xc, yc)

    return trainer, chunks


def predict_scores_from_pipeline(pipeline, X):
    scaler = pipeline.named_steps["scale"]
    model = pipeline.named_steps["model"]
    Xt = scaler.transform(X)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(Xt)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1]
        return proba.ravel()

    return model.predict(Xt).astype(float)


def print_history(title, history, limit=8):
    print(f"\n{title}")
    print("chunk | size | chunk_acc | cum_acc | memory_bytes")
    print("-" * 52)

    for row in history[:limit]:
        print(
            f"{row['chunk_index']:5d} | "
            f"{row['chunk_size']:4d} | "
            f"{row['chunk_accuracy']:.4f}   | "
            f"{row['cumulative_accuracy']:.4f}  | "
            f"{row['memory_bytes']}"
        )

    if len(history) > limit:
        print("...")
        last = history[-1]
        print(
            f"{last['chunk_index']:5d} | "
            f"{last['chunk_size']:4d} | "
            f"{last['chunk_accuracy']:.4f}   | "
            f"{last['cumulative_accuracy']:.4f}  | "
            f"{last['memory_bytes']}"
        )


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

    tree_trainer, tree_chunks = run_streaming_experiment(tree_model, X, y, chunk_size=chunk_size)
    forest_trainer, _ = run_streaming_experiment(forest_model, X, y, chunk_size=chunk_size)

    tree_cum = history_values(tree_trainer.history_, "cumulative_accuracy")
    forest_cum = history_values(forest_trainer.history_, "cumulative_accuracy")
    tree_chunk_acc = history_values(tree_trainer.history_, "chunk_accuracy")
    forest_chunk_acc = history_values(forest_trainer.history_, "chunk_accuracy")
    forest_memory = history_values(forest_trainer.history_, "memory_bytes")

    last_X, last_y = tree_chunks[-1]
    last_pred = forest_trainer.predict(last_X)
    full_scores = predict_scores_from_pipeline(forest_trainer.pipeline, X)
    full_pred = forest_trainer.predict(X)

    print("\nStreaming run completed")
    print(f"Chunks processed: {len(tree_trainer.history_)}")
    print(f"Tree final cumulative accuracy: {tree_cum[-1]:.4f}")
    print(f"Forest final cumulative accuracy: {forest_cum[-1]:.4f}")

    print_history("Tree history", tree_trainer.history_)
    print_history("Forest history", forest_trainer.history_)

    out_dir = Path(__file__).resolve().parent / "demo_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = compare_models(
        tree_cum,
        forest_cum,
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
        full_scores,
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
        "tree_final_cumulative_accuracy": tree_cum[-1],
        "forest_final_cumulative_accuracy": forest_cum[-1],
        "tree_history": tree_trainer.history_,
        "forest_history": forest_trainer.history_,
    }

    with (out_dir / "stream_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=json_default)

    print(f"\nSaved demo outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()