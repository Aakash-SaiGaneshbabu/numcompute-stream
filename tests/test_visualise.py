import matplotlib
matplotlib.use("Agg")

import numpy as np
from pathlib import Path

from numcompute.visualise import (
    plot_metric_over_time,
    compare_models,
    plot_predictions_vs_ground_truth,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_streaming_metric,
    plot_memory_usage,
)


def test_plot_metric_over_time(tmp_path):
    path = tmp_path / "metric.png"
    fig = plot_metric_over_time([0.2, 0.4, 0.6], save_path=path, show=False)

    assert path.exists()
    assert fig is not None
    assert len(fig.axes) == 1


def test_compare_models(tmp_path):
    path = tmp_path / "compare.png"
    fig = compare_models([0.1, 0.2], [0.3, 0.4], save_path=path, show=False)

    assert path.exists()
    assert fig is not None


def test_plot_predictions_vs_ground_truth(tmp_path):
    path = tmp_path / "pred.png"
    fig = plot_predictions_vs_ground_truth([0, 1, 1], [0, 1, 0], save_path=path, show=False)

    assert path.exists()
    assert fig is not None


def test_plot_confusion_matrix(tmp_path):
    path = tmp_path / "cm.png"
    fig = plot_confusion_matrix([0, 0, 1, 1], [0, 1, 0, 1], save_path=path, show=False)

    assert path.exists()
    assert fig is not None


def test_plot_roc_curve(tmp_path):
    path = tmp_path / "roc.png"
    fig = plot_roc_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], save_path=path, show=False)

    assert path.exists()
    assert fig is not None


def test_plot_streaming_metric(tmp_path):
    path = tmp_path / "stream_metric.png"
    fig = plot_streaming_metric([0.5, 0.6, 0.7], metric_name="Accuracy", save_path=path, show=False)

    assert path.exists()
    assert fig is not None


def test_plot_memory_usage(tmp_path):
    path = tmp_path / "memory.png"
    fig = plot_memory_usage([100, 200, 150], save_path=path, show=False)

    assert path.exists()
    assert fig is not None
