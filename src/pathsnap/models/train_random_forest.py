#!/usr/bin/env python3
"""Train and evaluate Random Forest candidate scorer via the shared harness (Milestone 3).

Usage:
    python -m pathsnap.models.train_random_forest
    python -m pathsnap.models.train_random_forest --max-depth 12 --name random_forest_d12
    python -m pathsnap.models.train_random_forest --max-depth 18 --name random_forest_d18

Model: Random Forest Classifier on Milestone 2 candidate feature table (36 features).
Imbalance: class_weight="balanced_subsample" dynamically adjusts weights per bootstrap sample.
Grouped 5-fold CV by trip_id, full fit on train trips, and evaluation on student_test.
Logs everything to outputs/results/experiments.csv and artifacts/models/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from pathsnap.models import harness as H


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "estimator": "sklearn.ensemble.RandomForestClassifier",
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "max_features": args.max_features,
        "class_weight": args.class_weight,
        "criterion": "gini",
        "n_jobs": args.n_jobs,
        "random_state": H.SEED,
        "preprocessing": "median impute + missing indicators (scale=False in Pipeline)",
        "threshold": 0.5,
    }


def plot_feature_importances(model_pipeline: Any, feature_names: list[str], output_path: Path) -> pd.DataFrame:
    """Extract and plot Gini feature importances from fitted Random Forest pipeline."""
    rf = model_pipeline.named_steps["model"]
    importances = rf.feature_importances_

    # Account for SimpleImputer missing indicators if present
    imputer = model_pipeline.named_steps["impute"]
    names = list(feature_names)
    if hasattr(imputer, "indicator_") and imputer.indicator_ is not None:
        missing_indices = imputer.indicator_.features_
        missing_names = [f"{names[i]}_missing" for i in missing_indices]
        all_feature_names = names + missing_names
    else:
        all_feature_names = names

    # Truncate or match names to importances length
    if len(all_feature_names) != len(importances):
        all_feature_names = [f"feat_{i}" for i in range(len(importances))]

    df_imp = pd.DataFrame({
        "feature": all_feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 10))
    plt.barh(df_imp["feature"][-25:], df_imp["importance"][-25:], color="steelblue")
    plt.xlabel("Gini Feature Importance")
    plt.title("Random Forest - Top 25 Feature Importances")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    csv_path = output_path.with_suffix(".csv")
    df_imp.sort_values("importance", ascending=False).to_csv(csv_path, index=False)
    print(f"Saved feature importances plot to {output_path} and CSV to {csv_path}")
    return df_imp


def model_factory(config: Dict[str, Any] | None = None) -> Any:
    cfg = {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_leaf": 10,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
        "criterion": "gini",
        "n_jobs": -1,
        "random_state": H.SEED,
    }
    if config:
        cfg.update(config)
    est = RandomForestClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        min_samples_leaf=cfg["min_samples_leaf"],
        max_features=cfg["max_features"],
        class_weight=cfg["class_weight"],
        criterion=cfg["criterion"],
        n_jobs=cfg["n_jobs"],
        random_state=cfg["random_state"],
    )
    return H.make_pipeline(est, scale=False)


def run_rf(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = build_config(args)

    def factory():
        return model_factory(cfg)

    print(f"\n=======================================================")
    print(f"Running Random Forest Experiment: {args.name}")
    print(f"Config: {json.dumps(cfg, indent=2)}")
    print(f"=======================================================\n")

    summary = H.run_experiment(factory, name=args.name, family="tree", config=cfg)

    # Plot feature importances using fitted model saved in artifacts
    model_path = Path("artifacts/models") / f"{args.name}.joblib"
    if model_path.exists():
        import joblib
        fitted_model = joblib.load(model_path)
        plot_path = Path("outputs/plots") / f"{args.name}_feature_importances.png"
        plot_feature_importances(fitted_model, H.FEATURES, plot_path)

    keys = [
        "macro_f1",
        "top1_when_candidate",
        "top1_all_points",
        "candidate_miss_rate",
        "route_continuity",
        "log_loss",
        "precision",
        "recall",
        "accuracy",
    ]
    print(f"\n{'metric':24s} {'fold-train':>10s} {'CV mean':>9s} {'CV std':>8s} {'full-train':>10s} {'student':>9s}")
    for k in keys:
        print(
            f"{k:24s} {summary[f'cv_train_{k}_mean']:10.4f} {summary[f'cv_{k}_mean']:9.4f} {summary[f'cv_{k}_std']:8.4f} "
            f"{summary[f'full_train_{k}']:10.4f} {summary[f'student_{k}']:9.4f}"
        )
    print(f"\nStudent confusion matrix [[TN, FP], [FN, TP]]: {summary['student_confusion_matrix']}")
    print(
        f"Timing: CV fit {summary['cv_fit_time_s_mean']:.1f}s/fold, full fit {summary['full_fit_time_s']:.1f}s, "
        f"inference {summary['student_infer_ms_per_1k_rows']:.2f} ms per 1k rows, largest trip: {summary['largest_trip']}"
    )

    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="random_forest")
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--max-depth", type=int, default=15)
    ap.add_argument("--min-samples-leaf", type=int, default=10)
    ap.add_argument("--max-features", default="sqrt")
    ap.add_argument("--class-weight", default="balanced_subsample")
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()

    run_rf(args)


if __name__ == "__main__":
    main()
