"""Unit tests for experiment logging."""

from pathlib import Path
import pandas as pd
from pathsnap.utils.experiment_log import log_experiment


def test_log_experiment(tmp_path: Path):
    csv_file = tmp_path / "experiments.csv"

    # Log two experimental runs
    log_experiment(
        model="baseline_viterbi",
        family="hmm",
        config={"sigma": 10.0, "beta": 2.0},
        metrics={"accuracy": 0.88, "precision": 0.85},
        seed=42,
        train_time_sec=1.23,
        inference_time_sec=0.45,
        notes="initial baseline",
        csv_path=csv_file,
    )

    log_experiment(
        model="lightgbm_scorer",
        family="lightgbm",
        config={"learning_rate": 0.05, "n_estimators": 100},
        metrics={"accuracy": 0.93, "precision": 0.91},
        seed=42,
        train_time_sec=5.41,
        inference_time_sec=0.12,
        notes="boosted candidates",
        csv_path=csv_file,
    )

    assert csv_file.exists()
    df = pd.read_csv(csv_file)
    assert len(df) == 2
    assert list(df["model"]) == ["baseline_viterbi", "lightgbm_scorer"]
    assert list(df["family"]) == ["hmm", "lightgbm"]
    assert "accuracy" in df["metrics_json"].iloc[0]
