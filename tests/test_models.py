"""Unit tests for Random Forest and KNN model pipelines and utilities."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from pathsnap.models.common import FEATURE_COLUMNS
from pathsnap.models.train_knn import (
    KNN_CONFIG,
    model_factory as knn_model_factory,
    subsample_training_fold,
)
from pathsnap.models.train_random_forest import (
    RF_CONFIG,
    model_factory as rf_model_factory,
)


@pytest.fixture
def synthetic_data():
    """Create small synthetic feature matrix and binary labels with intentional NaNs."""
    rng = np.random.default_rng(42)
    n = 100
    X = pd.DataFrame(
        rng.normal(size=(n, len(FEATURE_COLUMNS))),
        columns=FEATURE_COLUMNS,
    )
    # Inject NaNs into heading columns as observed in real stationary GPS data
    X.loc[:5, "gps_heading_deg"] = np.nan
    X.loc[:5, "bearing_diff_deg"] = np.nan
    X.loc[:5, "bearing_diff_undirected_deg"] = np.nan

    # Imbalanced labels: ~15% positive
    y = pd.Series((rng.random(n) < 0.15).astype(int))
    # Guarantee at least one positive
    y.iloc[0] = 1
    return X, y


def test_rf_model_factory():
    pipe = rf_model_factory()
    assert isinstance(pipe, Pipeline)
    assert "rf" in pipe.named_steps
    assert pipe.named_steps["rf"].n_estimators == RF_CONFIG["n_estimators"]
    assert pipe.named_steps["rf"].max_depth == RF_CONFIG["max_depth"]
    assert pipe.named_steps["rf"].min_samples_leaf == RF_CONFIG["min_samples_leaf"]

    # Test override
    pipe_custom = rf_model_factory({"n_estimators": 50, "max_depth": 8})
    assert pipe_custom.named_steps["rf"].n_estimators == 50
    assert pipe_custom.named_steps["rf"].max_depth == 8


def test_knn_model_factory():
    pipe = knn_model_factory()
    assert isinstance(pipe, Pipeline)
    assert "scaler" in pipe.named_steps
    assert "knn" in pipe.named_steps
    assert pipe.named_steps["knn"].n_neighbors == KNN_CONFIG["n_neighbors"]
    assert pipe.named_steps["knn"].weights == "distance"

    # Test override
    pipe_custom = knn_model_factory({"n_neighbors": 11, "weights": "uniform"})
    assert pipe_custom.named_steps["knn"].n_neighbors == 11
    assert pipe_custom.named_steps["knn"].weights == "uniform"


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


def test_knn_subsampling():
    n_pos = 20
    n_neg = 100
    y = pd.Series([1] * n_pos + [0] * n_neg)
    X = pd.DataFrame(np.zeros((len(y), 5)))

    # Subsample with ratio 3.0 (expect 20 pos + 60 neg = 80 rows)
    X_sub, y_sub = subsample_training_fold(X, y, negative_ratio=3.0, random_state=42)
    assert (y_sub == 1).sum() == n_pos
    assert (y_sub == 0).sum() == 60
    assert len(X_sub) == 80
