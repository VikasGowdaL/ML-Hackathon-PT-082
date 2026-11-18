#!/usr/bin/env python3
"""Evaluate the Bayesian-tuned LightGBM through the shared harness (tuned vs untuned comparison).

Usage:
    python -m pathsnap.models.train_lightgbm_tuned

Reads the best parameters found by ``bayesian_tune_lightgbm.py`` (outputs/results/lightgbm_best_params.json)
and runs exactly the same protocol as the untuned ``train_lightgbm.py`` baseline: grouped 5-fold CV on the 80
train trips, fit on all train trips, student-test scoring, logging to outputs/results/experiments.csv as
``lightgbm_tuned``. Everything else (class_weight="balanced", seed, n_jobs=4, preprocessing) is identical, so
the two rows are directly comparable.
"""

from __future__ import annotations

import json

from lightgbm import LGBMClassifier

from pathsnap.models import harness as H
from pathsnap.utils.config import get_repo_root


def main() -> None:
    best = json.loads((get_repo_root() / "outputs" / "results" / "lightgbm_best_params.json").read_text(encoding="utf-8"))
    params = best["parameters"]
    config = {
        "estimator": "lightgbm.LGBMClassifier",
        **params,
        "class_weight": "balanced",
        "random_state": H.SEED,
        "n_jobs": 4,
        "source": "Optuna TPE best trial (outputs/results/lightgbm_best_params.json)",
        "preprocessing": "median imputation + missing indicators",
        "loss": "binary log loss",
        "threshold": 0.5,
    }

    def factory():
        est = LGBMClassifier(**params, class_weight="balanced", random_state=H.SEED, n_jobs=4, verbosity=-1)
        return H.make_pipeline(est, scale=False)

    print(json.dumps(config, indent=2))
    s = H.run_experiment(factory, name="lightgbm_tuned", family="gradient_boosting", config=config,
                         notes="Bayesian-tuned LightGBM (same protocol as lightgbm baseline)")
    keys = ["macro_f1", "top1_when_candidate", "top1_all_points", "route_continuity", "log_loss", "precision", "recall"]
    print(f"\n{'metric':22s} {'fold-train':>10s} {'CV mean':>9s} {'CV std':>8s} {'student':>9s}")
    for k in keys:
        print(f"{k:22s} {s[f'cv_train_{k}_mean']:10.4f} {s[f'cv_{k}_mean']:9.4f} {s[f'cv_{k}_std']:8.4f} {s[f'student_{k}']:9.4f}")
    print(f"inference {s['student_infer_ms_per_1k_rows']:.2f} ms per 1k rows; full fit {s['full_fit_time_s']:.1f}s")


if __name__ == "__main__":
    main()
