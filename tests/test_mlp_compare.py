"""Tests for the MLP wrapper and the model comparison (Milestone 3, Person 4)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pathsnap.models.compare import build_comparison, key_config, load_runs
from pathsnap.models.train_mlp import EpochMLP


def test_epoch_mlp_history_and_early_stopping():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 4)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 1.2).astype(int)  # ~15 % positives
    m = EpochMLP(hidden_layer_sizes=(8,), max_epochs=15, patience=3, batch_size=64, train_loss_sample=1000).fit(X, y)
    h = pd.DataFrame(m.history_)
    assert list(h.epoch) == list(range(1, len(h) + 1)) and len(h) == m.n_epochs_ <= 15
    assert {"train_loss", "val_loss", "train_loss_unweighted", "val_loss_unweighted"} <= set(h.columns)
    assert np.isfinite(h[["train_loss", "val_loss"]].to_numpy()).all()
    assert h.val_loss.min() == pytest.approx(h.loc[h.epoch == m.best_epoch_, "val_loss"].item())
    p = m.predict_proba(X)
    assert p.shape == (3000, 2) and np.allclose(p.sum(axis=1), 1)
    assert m.split_["val_rows"] == 300  # last 10 % held out


def test_epoch_mlp_reproducible():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(1000, 3)).astype(np.float32)
    y = (X[:, 0] > 1).astype(int)
    a = EpochMLP(hidden_layer_sizes=(4,), max_epochs=3, batch_size=64, random_state=7).fit(X, y).predict_proba(X)
    b = EpochMLP(hidden_layer_sizes=(4,), max_epochs=3, batch_size=64, random_state=7).fit(X, y).predict_proba(X)
    assert np.allclose(a, b)


def test_comparison_from_log(tmp_path):
    metrics = {k: 0.5 for k in ["cv_macro_f1_mean", "cv_macro_f1_std", "cv_top1_when_candidate_mean",
                                "cv_top1_all_points_mean", "cv_route_continuity_mean", "cv_log_loss_mean",
                                "cv_train_macro_f1_mean", "student_macro_f1", "student_top1_when_candidate",
                                "student_top1_all_points", "student_route_continuity", "student_log_loss",
                                "full_fit_time_s", "student_infer_ms_per_1k_rows"]}
    metrics["student_confusion_matrix"] = [[9, 1], [2, 8]]
    rows = [
        ["2026-01-01T00:00:00+00:00", "old_run", "hmm", "", '{"a": 1}', '{"acc": 1.0}', "", "", ""],
        ["2026-01-01T00:00:00+00:00", "lr", "linear", 42, json.dumps({"C": 1.0, "loss": "x"}), json.dumps({**metrics, "cv_macro_f1_mean": 0.6}), 1, 1, ""],
        ["2026-01-02T00:00:00+00:00", "lr", "linear", 42, json.dumps({"C": 0.5}), json.dumps(metrics), 1, 1, ""],
    ]
    cols = ["timestamp", "model", "family", "seed", "config_json", "metrics_json", "train_time_sec", "inference_time_sec", "notes"]
    p = tmp_path / "experiments.csv"
    pd.DataFrame(rows, columns=cols).to_csv(p, index=False)
    comp = build_comparison(load_runs(p))
    assert list(comp.model) == ["lr", "old_run"]                 # latest run per model; harness runs first
    lr = comp[comp.model == "lr"].iloc[0]
    assert lr.cv_macro_f1_mean == 0.5 and lr.key_config == "C=0.5" and lr.na_reason == ""
    old = comp[comp.model == "old_run"].iloc[0]
    assert old.cv_macro_f1_mean == "N/A" and "not produced by the Milestone 3 harness" in old.na_reason
    assert key_config({"C": 1, "loss": "x", "hidden_layer_sizes": [64, 32]}) == "C=1; hidden_layer_sizes=[64, 32]"
