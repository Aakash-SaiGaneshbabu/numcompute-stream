import numpy as np
import pytest

from numcompute.stream import StreamTrainer, iter_chunks
from numcompute.metrics import StreamingAccuracy


class DummyModel:
    def __init__(self):
        self.partial_fit_calls = 0
        self.fit_calls = 0

    def partial_fit(self, X, y):
        self.partial_fit_calls += 1
        return self

    def predict(self, X):
        return np.zeros(X.shape[0], dtype=int)


class DummyPipeline:
    def __init__(self):
        self.partial_fit_calls = 0

    def partial_fit(self, X, y):
        self.partial_fit_calls += 1
        return self

    def predict(self, X):
        return np.ones(X.shape[0], dtype=int)


def test_stream_trainer_with_model():
    model = DummyModel()
    trainer = StreamTrainer(model=model)

    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0, 0])

    trainer.fit_chunk(X, y)
    entry = trainer.score_chunk(X, y)

    assert model.partial_fit_calls == 1
    assert entry["chunk_index"] == 1
    assert entry["chunk_size"] == 2
    assert entry["chunk_accuracy"] == 1.0
    assert "memory_bytes" in entry
    assert trainer.result() == 1.0


def test_stream_trainer_with_pipeline():
    pipe = DummyPipeline()
    trainer = StreamTrainer(pipeline=pipe, metric_tracker=StreamingAccuracy())

    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([1, 1])

    trainer.fit_chunk(X, y)
    entry = trainer.score_chunk(X, y)

    assert pipe.partial_fit_calls == 1
    assert entry["chunk_accuracy"] == 1.0
    assert trainer.result() == 1.0


def test_stream_trainer_reset():
    model = DummyModel()
    trainer = StreamTrainer(model=model)

    X = np.array([[1.0, 2.0]])
    y = np.array([0])

    trainer.fit_chunk(X, y)
    trainer.score_chunk(X, y)
    trainer.reset()

    assert trainer.history_ == []
    assert trainer.chunk_index_ == 0


def test_stream_trainer_requires_model_or_pipeline():
    with pytest.raises(ValueError):
        StreamTrainer()


def test_iter_chunks_with_labels():
    X = np.arange(12).reshape(6, 2)
    y = np.array([0, 1, 2, 3, 4, 5])

    chunks = list(iter_chunks(X, y, chunk_size=2))

    assert len(chunks) == 3
    assert np.array_equal(chunks[0][0], X[:2])
    assert np.array_equal(chunks[0][1], y[:2])


def test_iter_chunks_drop_last():
    X = np.arange(10).reshape(5, 2)

    chunks = list(iter_chunks(X, chunk_size=2, drop_last=True))

    assert len(chunks) == 2
    assert np.array_equal(chunks[0], X[:2])
    assert np.array_equal(chunks[1], X[2:4])


def test_iter_chunks_rejects_invalid_input():
    with pytest.raises(ValueError):
        list(iter_chunks(np.array([1, 2, 3]), chunk_size=2))