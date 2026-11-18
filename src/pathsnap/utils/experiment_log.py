"""Experiment logging helper for PathSnap.

Appends experiment runs to outputs/results/experiments.csv in a thread-safe,
cross-platform manner.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

from pathsnap.utils.config import get_path, get_repo_root

EXPERIMENT_CSV_HEADERS = [
    "timestamp",
    "model",
    "family",
    "seed",
    "config_json",
    "metrics_json",
    "train_time_sec",
    "inference_time_sec",
    "notes",
]


def log_experiment(
    model: str,
    family: str,
    config: Dict[str, Any] | str,
    metrics: Dict[str, Any] | str,
    seed: Optional[int] = None,
    train_time_sec: Optional[float] = None,
    inference_time_sec: Optional[float] = None,
    notes: str = "",
    csv_path: Optional[Union[Path, str]] = None,
) -> Path:
    """Append a row to the experiment tracking CSV.

    Args:
        model: Identifier/name of the specific model (e.g. 'viterbi_baseline_v1').
        family: High-level model category (e.g. 'hmm', 'lightgbm', 'xgboost', 'heuristic').
        config: Dict or JSON string of hyperparameters/configurations.
        metrics: Dict or JSON string of evaluation metrics (e.g. {'arc_accuracy': 0.89}).
        seed: Random seed used (if applicable).
        train_time_sec: Training / fitting time in seconds.
        inference_time_sec: Evaluation / inference runtime in seconds.
        notes: Optional free-form notes or comments.
        csv_path: Custom CSV path, defaults to outputs/results/experiments.csv.

    Returns:
        Path to the experiments CSV file.
    """
    if csv_path is None:
        try:
            target_file = get_path("experiments_log", default="outputs/results/experiments.csv")
        except Exception:
            target_file = get_repo_root() / "outputs" / "results" / "experiments.csv"
    else:
        target_file = Path(csv_path)
        if not target_file.is_absolute():
            target_file = (get_repo_root() / target_file).resolve()

    target_file.parent.mkdir(parents=True, exist_ok=True)

    config_str = config if isinstance(config, str) else json.dumps(config, sort_keys=True)
    metrics_str = metrics if isinstance(metrics, str) else json.dumps(metrics, sort_keys=True)
    current_time = datetime.now(timezone.utc).isoformat()

    file_exists = target_file.exists() and target_file.stat().st_size > 0

    with open(target_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(EXPERIMENT_CSV_HEADERS)
        writer.writerow([
            current_time,
            model,
            family,
            seed if seed is not None else "",
            config_str,
            metrics_str,
            f"{train_time_sec:.4f}" if train_time_sec is not None else "",
            f"{inference_time_sec:.4f}" if inference_time_sec is not None else "",
            notes,
        ])

    return target_file
