import numpy as np

from numcompute.pipeline import Pipeline, FeatureUnion
from numcompute.preprocessing import StandardScaler


class DummyEstimator:
    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class DummyTransformer:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X) * 2


def test_pipeline_fit_predict():
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", DummyEstimator()),
    ])

    X = np.array([[1], [2], [3]])

    pipe.fit(X)

    pred = pipe.predict(X)

    assert len(pred) == 3


def test_pipeline_fit_transform():
    pipe = Pipeline([
        ("double", DummyTransformer()),
        ("double2", DummyTransformer()),
    ])

    X = np.array([[1], [2]])

    result = pipe.fit_transform(X)

    assert np.array_equal(result, np.array([[4], [8]]))


def test_feature_union():
    union = FeatureUnion([
        ("t1", DummyTransformer()),
        ("t2", DummyTransformer()),
    ])

    X = np.array([[1], [2]])

    result = union.fit_transform(X)

    assert result.shape == (2, 2)