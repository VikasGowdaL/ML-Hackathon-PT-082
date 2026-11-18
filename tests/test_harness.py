"""Tests for the shared model harness (Milestone 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from pathsnap.models import harness as H


@pytest.fixture
def toy(monkeypatch):
    """Two observations of trip 'a' and one of trip 'b'; arcs 0->1->2 in 'a'."""
    df = pd.DataFrame({
        "trip_id": ["a", "a", "a", "a", "b", "b"],
        "obs_idx": [0, 0, 1, 1, 0, 0],
        "arc_id": [10, 11, 12, 13, 0, 1],
        "segment_id": [0] * 6,
        "label": [1, 0, 1, 0, 0, 0],   # trip b: reference arc not among candidates
    })
    # trip a: arc 10 = 0->1, arc 12 = 1->2 (connected); trip b arcs irrelevant
    f = np.zeros(20, dtype=int); t = np.zeros(20, dtype=int)
    f[10], t[10], f[12], t[12], f[11], t[11], f[13], t[13] = 0, 1, 1, 2, 5, 6, 7, 8
    monkeypatch.setitem(H._ARC_CACHE, "a", (f, t))
    monkeypatch.setitem(H._ARC_CACHE, "b", (np.array([0, 3]), np.array([1, 4])))
    counts = {"a": {"observations": 3}, "b": {"observations": 1}}  # trip a has 1 point without candidates
    return df, counts


def test_observation_metrics(toy):
    df, counts = toy
    p = np.array([0.9, 0.2, 0.8, 0.1, 0.6, 0.3])
    m = H.compute_metrics(df, p, counts)
    # top-1 picks arc 10, 12 (correct) and arc 0 in b (wrong, no reference candidate)
    assert m["n_observations"] == 4 and m["n_observations_with_reference_candidate"] == 2
    assert m["top1_all_points"] == pytest.approx(2 / 4)
    assert m["top1_when_candidate"] == pytest.approx(1.0)
    assert m["candidate_miss_rate"] == pytest.approx(0.5)
    assert m["route_continuity"] == pytest.approx(1.0)  # 10 -> 12 connected (1 -> 1)
    assert m["confusion_matrix"] == [[3, 1], [0, 2]]
    # swap the top-1 of obs 1 to an unconnected arc -> continuity 0
    m2 = H.compute_metrics(df, np.array([0.9, 0.2, 0.1, 0.8, 0.6, 0.3]), counts)
    assert m2["route_continuity"] == 0.0 and m2["top1_when_candidate"] == pytest.approx(0.5)


def test_pipeline_fits_preprocessing_inside():
    rng = np.random.default_rng(0)
    X = rng.normal(5, 2, size=(200, 3)); X[::7, 1] = np.nan
    y = (X[:, 0] > 5).astype(int)
    pipe = H.make_pipeline(LogisticRegression(), scale=True)
    pipe.fit(X[:100], y[:100])
    # scaler statistics come from the training rows only (after imputation + indicator)
    imputed = pipe.named_steps["impute"].transform(X[:100])
    assert np.allclose(pipe.named_steps["scale"].mean_, imputed.mean(axis=0))
    assert pipe.predict_proba(X[100:]).shape == (100, 2)
    assert "scale" not in H.make_pipeline(LogisticRegression(), scale=False).named_steps


def test_groupkfold_trip_disjoint():
    groups = np.repeat([f"t{i}" for i in range(10)], 5)
    for tr, va in GroupKFold(n_splits=H.N_SPLITS).split(groups, groups, groups):
        assert not set(groups[tr]) & set(groups[va])


def test_feature_set():
    assert len(H.FEATURES) == len(set(H.FEATURES))
    for c in H.EXCLUDED_FEATURES:
        assert c not in H.FEATURES
    for c in ("label", "reference_arc_id", "trip_id", "obs_idx", "arc_id"):
        assert c not in H.FEATURES


def test_real_tables_if_present():
    try:
        tr = H.load_train_table()
    except FileNotFoundError:
        pytest.skip("feature tables not built")
    assert (tr.partition == "train").all() and tr[H.GROUP].nunique() == 80
    assert not tr.duplicated(["trip_id", "obs_idx", "arc_id"]).any()
