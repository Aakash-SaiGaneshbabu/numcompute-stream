import numpy as np
import pytest

from numcompute.metrics import (
    accuracy,
    precision,
    recall,
    f1,
    confusion_matrix,
    mse,
    roc_curve,
    auc,
    StreamingAccuracy,
    StreamingPrecision,
    StreamingRecall,
    StreamingF1,
    StreamingConfusionMatrix,
    StreamingMSE,
    StreamingAUC,
)


def test_classification_metrics():
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])

    assert accuracy(y_true, y_pred) == 0.75
    assert precision(y_true, y_pred) == 1.0
    assert recall(y_true, y_pred) == 2 / 3
    assert f1(y_true, y_pred) == pytest.approx(0.8)


def test_confusion_matrix():
    y_true = [1, 0, 1, 1]
    y_pred = [1, 0, 0, 1]

    cm = confusion_matrix(y_true, y_pred)

    assert np.array_equal(
        cm,
        np.array([[1, 0],
                  [1, 2]])
    )


def test_mse():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 4])

    assert mse(y_true, y_pred) == pytest.approx(1 / 3)


def test_roc_auc():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])

    fpr, tpr, _ = roc_curve(y_true, y_score)

    assert len(fpr) == 4
    assert len(tpr) == 4

    area = auc(fpr, tpr)
    assert area >= 0


def test_streaming_accuracy():
    metric = StreamingAccuracy()

    metric.update([1, 0], [1, 1])
    metric.update([1, 0], [1, 0])

    assert metric.result() == 0.75


def test_streaming_precision():
    metric = StreamingPrecision()

    metric.update([1, 0], [1, 1])

    assert metric.result() == 0.5


def test_streaming_recall():
    metric = StreamingRecall()

    metric.update([1, 0, 1], [1, 0, 0])

    assert metric.result() == 0.5


def test_streaming_f1():
    metric = StreamingF1()

    metric.update([1, 0, 1], [1, 0, 0])

    assert metric.result() > 0


def test_streaming_confusion_matrix():
    metric = StreamingConfusionMatrix()

    metric.update([1, 0], [1, 1])

    cm = metric.result()

    assert cm.shape == (2, 2)


def test_streaming_mse():
    metric = StreamingMSE()

    metric.update([1, 2, 3], [1, 2, 4])

    assert metric.result() == pytest.approx(1 / 3)


def test_streaming_auc():
    metric = StreamingAUC()

    metric.update(
        [0, 0, 1, 1],
        [0.1, 0.4, 0.35, 0.8]
    )

    result = metric.result()

    assert not np.isnan(result)