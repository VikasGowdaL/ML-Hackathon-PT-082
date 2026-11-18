"""Random Forest model for PathSnap candidate arc scoring.

Scores each (GPS observation, candidate arc) pair with predict_proba.
Exposes:
  - RF_CONFIG: default hyperparameters in one central place
  - model_factory(): returns an sklearn Pipeline
  - run(): entry point for the shared harness, containing a temporary smoke test
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from pathsnap.models.common import FEATURE_COLUMNS, load_dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Central Hyperparameters Configuration
# ---------------------------------------------------------------------------
# Justification:
# - n_estimators=200: Provides ensemble variance reduction and stable class
#   probabilities without prohibitive training latency on ~1.77M candidate pairs.
# - max_depth=15: Prevents deep tree splits that overfit local GPS measurement
#   noise and isolated geometric artifacts; limits leaf memory usage.
# - min_samples_leaf=10: Ensures every leaf has sufficient observations to
#   estimate calibrated local probabilities; prevents 1-sample leaves.
# - class_weight="balanced_subsample": Adjusts split criteria dynamically per
#   bootstrap sample to account for the ~8.57:1 negative-to-positive class
#   imbalance without throwing away training samples.
# - n_jobs=-1: Maximizes multi-core CPU utilization during tree construction.
# - random_state=42: Guarantees strict reproducibility.
RF_CONFIG: Dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_leaf": 10,
    "class_weight": "balanced_subsample",
    "n_jobs": -1,
    "random_state": 42,
    "impute_missing": True,
}

# Temporary trips used for smoke testing before harness integration
TEMPORARY_SMOKE_TEST_TRIPS = ["00000039", "00000005", "00000045"]


def model_factory(config: Optional[Dict[str, Any]] = None) -> Pipeline:
    """
    Construct an sklearn Pipeline for Random Forest scoring.

    Parameters:
        config: Optional dictionary overriding keys in RF_CONFIG.

    Returns:
        sklearn Pipeline (SimpleImputer + RandomForestClassifier).
        No scaling is applied since tree-based models are scale-invariant.
    """
    cfg = dict(RF_CONFIG)
    if config:
        cfg.update(config)

    steps = []
    if cfg.get("impute_missing", True):
        # Median imputer handles 0.10% NaNs in heading features (stationary points)
        steps.append(("imputer", SimpleImputer(strategy="median")))

    steps.append((
        "rf",
        RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            min_samples_leaf=cfg["min_samples_leaf"],
            class_weight=cfg["class_weight"],
            n_jobs=cfg["n_jobs"],
            random_state=cfg["random_state"],
        ),
    ))

    return Pipeline(steps)


def run(
    trips: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
    radius_m: int = 50,
) -> Dict[str, Any]:
    """
    Single run() entry point designed to plug into the shared harness.

    Until the harness in src/pathsnap/models/harness.py is ready, this function
    executes a temporary self-contained smoke test on 3 small trips.

    Parameters:
        trips: Optional list of trip IDs. If None, runs the smoke test.
        config: Optional hyperparameter overrides.
        radius_m: Search radius in metres for candidate features (default: 50).

    Returns:
        Dict with pipeline, feature_names, and smoke test / run summary.
    """
    is_temporary_smoke_test = trips is None
    target_trips = TEMPORARY_SMOKE_TEST_TRIPS if is_temporary_smoke_test else trips

    print(
        f"[Random Forest] {'TEMPORARY SMOKE TEST' if is_temporary_smoke_test else 'Run'} "
        f"on {len(target_trips)} trips: {target_trips}"
    )

    t0 = time.time()
    X, y, meta = load_dataset(target_trips, partition="train", radius_m=radius_m, labeled=True)
    t_load = time.time() - t0

    print(
        f"  Loaded dataset: X shape {X.shape}, y distribution: "
        f"0={int((y == 0).sum())}, 1={int((y == 1).sum())} ({t_load:.2f}s)"
    )

    pipeline = model_factory(config)

    t0 = time.time()
    pipeline.fit(X, y)
    t_fit = time.time() - t0

    t0 = time.time()
    sample_size = min(1000, len(X))
    probs = pipeline.predict_proba(X.iloc[:sample_size])
    t_infer = time.time() - t0

    assert probs.shape == (sample_size, 2), f"Expected shape ({sample_size}, 2), got {probs.shape}"
    assert np.allclose(probs.sum(axis=1), 1.0), "Probabilities do not sum to 1.0"
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Probabilities out of [0, 1] range"

    result = {
        "status": "success",
        "is_temporary_smoke_test": is_temporary_smoke_test,
        "pipeline": pipeline,
        "feature_names": FEATURE_COLUMNS,
        "n_samples": len(X),
        "fit_time_s": t_fit,
        "inference_ms_per_1k": (t_infer / sample_size) * 1000.0 * 1000.0,
        "pos_class_prob_mean": float(probs[:, 1].mean()),
        "pos_class_prob_max": float(probs[:, 1].max()),
    }

    print(
        f"  Fit time: {t_fit:.2f}s | Inference latency: {result['inference_ms_per_1k']:.2f} ms / 1k rows"
    )
    print(
        f"  Predicted prob(class=1) - mean: {result['pos_class_prob_mean']:.4f}, max: {result['pos_class_prob_max']:.4f}"
    )
    print("  [Random Forest] Verification passed.")

    return result


if __name__ == "__main__":
    run()
