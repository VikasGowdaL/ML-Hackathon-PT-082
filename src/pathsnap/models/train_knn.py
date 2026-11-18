"""K-Nearest Neighbors (KNN) model for PathSnap candidate arc scoring.

Scores each (GPS observation, candidate arc) pair with predict_proba.
Exposes:
  - KNN_CONFIG: default hyperparameters in one central place
  - model_factory(): returns an sklearn Pipeline (StandardScaler + KNeighborsClassifier)
  - subsample_training_fold(): safe negative downsampler strictly for training folds
  - run(): entry point for the shared harness, containing a temporary smoke test
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pathsnap.models.common import FEATURE_COLUMNS, load_dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Central Hyperparameters Configuration
# ---------------------------------------------------------------------------
# Justification:
# - n_neighbors=21: A moderately large, odd k yields smooth, continuous probability
#   estimates (rather than coarse step-function fractions like k=3: 0, 0.33, 0.67, 1.0)
#   while remaining strictly local in standardized geometric space.
# - weights="distance": Inverse Euclidean distance weighting gives immediate spatial
#   and angular matches higher decision weight than distant boundary candidates.
# - metric="minkowski", p=2: Standard Euclidean metric on standardized features
#   ensures isotropic geometric distance across normalized coordinates and angles.
# - n_jobs=-1: Maximizes multi-core CPU utilization during neighbor searches.
# - subsample_negatives=True: Caps negative instances in training fold to keep
#   O(N_train * N_test) search complexity tractable on large datasets (~1.77M rows).
# - negative_subsample_ratio=4.0: Retains all positive instances while sampling 4
#   negatives per positive, balancing inference latency with decision boundary precision.
KNN_CONFIG: Dict[str, Any] = {
    "n_neighbors": 21,
    "weights": "distance",
    "metric": "minkowski",
    "p": 2,
    "n_jobs": -1,
    "impute_missing": True,
    "subsample_negatives": False,  # Default off for smoke tests; activated inside CV training fold
    "negative_subsample_ratio": 4.0,
    "max_train_samples": 50000,
    "random_state": 42,
}

# Temporary trips used for smoke testing before harness integration
TEMPORARY_SMOKE_TEST_TRIPS = ["00000039", "00000005", "00000045"]


def model_factory(config: Optional[Dict[str, Any]] = None) -> Pipeline:
    """
    Construct an sklearn Pipeline for KNN candidate scoring.

    Parameters:
        config: Optional dictionary overriding keys in KNN_CONFIG.

    Returns:
        sklearn Pipeline (SimpleImputer + StandardScaler + KNeighborsClassifier).
        StandardScaler is mandatory because KNN is highly scale-dependent.
    """
    cfg = dict(KNN_CONFIG)
    if config:
        cfg.update(config)

    steps = []
    if cfg.get("impute_missing", True):
        # Median imputer handles 0.10% NaNs in heading features (stationary points)
        steps.append(("imputer", SimpleImputer(strategy="median")))

    steps.append(("scaler", StandardScaler()))
    steps.append((
        "knn",
        KNeighborsClassifier(
            n_neighbors=cfg["n_neighbors"],
            weights=cfg["weights"],
            metric=cfg["metric"],
            p=cfg["p"],
            n_jobs=cfg["n_jobs"],
        ),
    ))

    return Pipeline(steps)


def subsample_training_fold(
    X: pd.DataFrame,
    y: pd.Series,
    negative_ratio: float = 4.0,
    max_samples: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Subsamples negative candidates inside the training fold ONLY.

    CRITICAL RULES:
    1. NEVER apply to validation or test folds. Validation must evaluate against 100% of candidates.
    2. Keeps 100% of positive (ground-truth) candidate arcs to prevent information loss.
    3. Randomly samples negatives up to (n_pos * negative_ratio) to reduce KD-tree / brute-force
       search complexity during inference.
    """
    pos_mask = (y == 1).to_numpy()
    neg_mask = (y == 0).to_numpy()

    pos_indices = np.where(pos_mask)[0]
    neg_indices = np.where(neg_mask)[0]

    n_pos = len(pos_indices)
    n_neg = len(neg_indices)

    target_n_neg = min(n_neg, int(n_pos * negative_ratio))

    rng = np.random.default_rng(random_state)
    kept_neg_indices = rng.choice(neg_indices, size=target_n_neg, replace=False)

    all_indices = np.concatenate([pos_indices, kept_neg_indices])
    rng.shuffle(all_indices)

    if max_samples is not None and len(all_indices) > max_samples:
        all_indices = all_indices[:max_samples]

    return X.iloc[all_indices].copy(), y.iloc[all_indices].copy()


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
        f"[KNN] {'TEMPORARY SMOKE TEST' if is_temporary_smoke_test else 'Run'} "
        f"on {len(target_trips)} trips: {target_trips}"
    )

    t0 = time.time()
    X, y, meta = load_dataset(target_trips, partition="train", radius_m=radius_m, labeled=True)
    t_load = time.time() - t0

    print(
        f"  Loaded dataset: X shape {X.shape}, y distribution: "
        f"0={int((y == 0).sum())}, 1={int((y == 1).sum())} ({t_load:.2f}s)"
    )

    cfg = dict(KNN_CONFIG)
    if config:
        cfg.update(config)

    # If subsampling requested for training fold
    if cfg.get("subsample_negatives", False):
        X_train, y_train = subsample_training_fold(
            X,
            y,
            negative_ratio=cfg.get("negative_subsample_ratio", 4.0),
            max_samples=cfg.get("max_train_samples"),
            random_state=cfg.get("random_state", 42),
        )
        print(f"  Training fold subsampled: {len(X)} -> {len(X_train)} rows")
    else:
        X_train, y_train = X, y

    pipeline = model_factory(cfg)

    t0 = time.time()
    pipeline.fit(X_train, y_train)
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
        "n_train_samples": len(X_train),
        "fit_time_s": t_fit,
        "inference_ms_per_1k": (t_infer / sample_size) * 1000.0 * 1000.0,
        "pos_class_prob_mean": float(probs[:, 1].mean()),
        "pos_class_prob_max": float(probs[:, 1].max()),
    }

    print(
        f"  Fit time: {t_fit:.3f}s | Inference latency: {result['inference_ms_per_1k']:.2f} ms / 1k rows"
    )
    print(
        f"  Predicted prob(class=1) - mean: {result['pos_class_prob_mean']:.4f}, max: {result['pos_class_prob_max']:.4f}"
    )
    print("  [KNN] Verification passed.")

    return result


if __name__ == "__main__":
    run()
