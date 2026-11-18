"""Unit tests for Random Forest and KNN model pipelines and utilities."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from pathsnap.models import harness as H
from pathsnap.models.train_knn import (
    SubsampledKNeighborsClassifier,
    model_factory as knn_model_factory,
)
from pathsnap.models.train_random_forest import (
    model_factory as rf_model_factory,
)


@pytest.fixture
def synthetic_data():
    """Create small synthetic feature matrix and binary labels with intentional NaNs."""
    rng = np.random.default_rng(42)
    n = 100
    X = pd.DataFrame(
        rng.normal(size=(n, len(H.FEATURES))),
        columns=H.FEATURES,
    )
    # Inject NaNs into heading columns as observed in real stationary GPS data
    if "bearing_diff_deg" in X.columns:
        X.loc[:5, "bearing_diff_deg"] = np.nan
    if "bearing_diff_undirected_deg" in X.columns:
        X.loc[:5, "bearing_diff_undirected_deg"] = np.nan

    # Imbalanced labels: ~15% positive
    y = pd.Series((rng.random(n) < 0.15).astype(int))
    y.iloc[0] = 1
    return X, y


def test_rf_model_factory():
    pipe = rf_model_factory()
    assert isinstance(pipe, Pipeline)
    assert "model" in pipe.named_steps
    assert pipe.named_steps["model"].n_estimators == 200
    assert pipe.named_steps["model"].max_depth == 15
    assert pipe.named_steps["model"].min_samples_leaf == 10

    # Test override
    pipe_custom = rf_model_factory({"n_estimators": 50, "max_depth": 8})
    assert pipe_custom.named_steps["model"].n_estimators == 50
    assert pipe_custom.named_steps["model"].max_depth == 8


def test_knn_model_factory():
    pipe = knn_model_factory()
    assert isinstance(pipe, Pipeline)
    assert "scale" in pipe.named_steps
    assert "model" in pipe.named_steps
    assert pipe.named_steps["model"].n_neighbors == 21
    assert pipe.named_steps["model"].weights == "distance"

    # Test override
    pipe_custom = knn_model_factory({"n_neighbors": 11, "weights": "uniform"})
    assert pipe_custom.named_steps["model"].n_neighbors == 11
    assert pipe_custom.named_steps["model"].weights == "uniform"


def test_rf_fit_predict_proba(synthetic_data):
    X, y = synthetic_data
    pipe = rf_model_factory({"n_estimators": 10, "random_state": 42})
    pipe.fit(X, y)

    probs = pipe.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_knn_fit_predict_proba(synthetic_data):
    X, y = synthetic_data
    pipe = knn_model_factory({"n_neighbors": 5})
    pipe.fit(X, y)

    probs = pipe.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_subsampled_knn_invariance():
    n_pos = 20
    n_neg = 100
    y = np.array([1] * n_pos + [0] * n_neg)
    X = np.zeros((len(y), 5))

    clf = SubsampledKNeighborsClassifier(
        n_neighbors=5,
        subsample_negatives=True,
        negative_subsample_ratio=3.0,
        random_state=42,
    )
    clf.fit(X, y)
    # Target negative = min(100, 20 * 3) = 60. Total samples in fitted estimator = 80
    assert len(clf.estimator_._fit_X) == 80
