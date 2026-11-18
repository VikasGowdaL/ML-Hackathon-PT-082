#!/usr/bin/env python3
"""Train and evaluate K-Nearest Neighbors (KNN) candidate scorer via the shared harness (Milestone 3).

Usage:
    python -m pathsnap.models.train_knn                                     # default initial run (k=21)
    python -m pathsnap.models.train_knn --k 11 --name knn_k11
    python -m pathsnap.models.train_knn --k 31 --name knn_k31
    python -m pathsnap.models.train_knn --sweep-k                           # manual sensitivity sweep over k

Model: Distance-weighted KNN on standardized features (median imputation + missing indicators + StandardScaler).
Loss / Decision: Weighted neighbor voting with inverse Euclidean distance: w_i = 1 / d(x, x_i).
Imbalance & Latency Feasibility:
  - Inside training folds only, retains 100% of positive instances and downsamples negatives (ratio 4:1,
    capped at 50,000 samples) to ensure inference latency is feasible (~38 ms/1k rows instead of ~5.5s/1k rows).
  - Validation and student_test folds are NEVER subsampled (evaluated on 100% of candidate observations).
Grouped 5-fold CV by trip_id, full fit on train trips, and evaluation on student_test.
Logs everything to outputs/results/experiments.csv and artifacts/models/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier

from pathsnap.models import harness as H


class SubsampledKNeighborsClassifier(BaseEstimator, ClassifierMixin):
    """
    KNeighborsClassifier with safe negative-candidate subsampling on the training fold ONLY.

    Invariance guarantees:
    - Retains 100% of minority positive instances (reference arcs).
    - Subsamples majority negative candidates to negative_subsample_ratio * n_pos.
    - Evaluation (predict_proba / score) is executed on the full, un-subsampled validation/test sets.
    """

    def __init__(
        self,
        n_neighbors: int = 21,
        weights: str = "distance",
        metric: str = "minkowski",
        p: int = 2,
        subsample_negatives: bool = True,
        negative_subsample_ratio: float = 4.0,
        max_train_samples: int = 50000,
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.p = p
        self.subsample_negatives = subsample_negatives
        self.negative_subsample_ratio = negative_subsample_ratio
        self.max_train_samples = max_train_samples
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> SubsampledKNeighborsClassifier:
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        if self.subsample_negatives:
            pos_idx = np.where(y_arr == 1)[0]
            neg_idx = np.where(y_arr == 0)[0]
            target_neg = min(len(neg_idx), int(len(pos_idx) * self.negative_subsample_ratio))

            rng = np.random.default_rng(self.random_state)
            kept_neg = rng.choice(neg_idx, size=target_neg, replace=False)

            idx = np.concatenate([pos_idx, kept_neg])
            rng.shuffle(idx)

            if self.max_train_samples and len(idx) > self.max_train_samples:
                idx = idx[: self.max_train_samples]

            X_fit = X_arr[idx]
            y_fit = y_arr[idx]
        else:
            X_fit = X_arr
            y_fit = y_arr

        self.estimator_ = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric,
            p=self.p,
            n_jobs=self.n_jobs,
        )
        self.estimator_.fit(X_fit, y_fit)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.estimator_.predict_proba(X)

    def predict(self, X: Any) -> np.ndarray:
        return self.estimator_.predict(X)


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "estimator": "pathsnap.models.train_knn.SubsampledKNeighborsClassifier",
        "n_neighbors": args.k,
        "weights": args.weights,
        "metric": args.metric,
        "p": 2,
        "subsample_negatives": args.subsample_negatives,
        "negative_subsample_ratio": args.negative_ratio,
        "max_train_samples": args.max_train_samples,
        "n_jobs": args.n_jobs,
        "random_state": H.SEED,
        "preprocessing": "median impute + missing indicators + StandardScaler (scale=True in Pipeline)",
        "threshold": 0.5,
    }


def model_factory(config: Dict[str, Any] | None = None) -> Any:
    cfg = {
        "n_neighbors": 21,
        "weights": "distance",
        "metric": "minkowski",
        "p": 2,
        "subsample_negatives": True,
        "negative_subsample_ratio": 4.0,
        "max_train_samples": 50000,
        "n_jobs": -1,
        "random_state": H.SEED,
    }
    if config:
        cfg.update(config)
    est = SubsampledKNeighborsClassifier(
        n_neighbors=cfg["n_neighbors"],
        weights=cfg["weights"],
        metric=cfg["metric"],
        p=cfg["p"],
        subsample_negatives=cfg["subsample_negatives"],
        negative_subsample_ratio=cfg["negative_subsample_ratio"],
        max_train_samples=cfg["max_train_samples"],
        n_jobs=cfg["n_jobs"],
        random_state=cfg["random_state"],
    )
    return H.make_pipeline(est, scale=True)


def run_single_knn(name: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    def factory():
        return model_factory(cfg)

    print(f"\n=======================================================")
    print(f"Running KNN Experiment: {name}")
    print(f"Config: {json.dumps(cfg, indent=2)}")
    print(f"=======================================================\n")

    summary = H.run_experiment(factory, name=name, family="instance_based", config=cfg)

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


def run_k_sweep(base_args: argparse.Namespace) -> None:
    """Run manual sensitivity sweep over k and plot CV Macro F1 curve."""
    k_values = [11, 21, 31]
    cv_scores = []
    print(f"\nStarting KNN k sensitivity sweep over {k_values}...")

    for k in k_values:
        run_name = f"knn_k{k}" if k != 21 else "knn"
        args_copy = argparse.Namespace(**vars(base_args))
        args_copy.k = k
        args_copy.name = run_name
        cfg = build_config(args_copy)
        summary = run_single_knn(run_name, cfg)
        cv_scores.append({
            "k": k,
            "cv_macro_f1": summary["cv_macro_f1_mean"],
            "cv_macro_f1_std": summary["cv_macro_f1_std"],
            "cv_top1": summary["cv_top1_when_candidate_mean"],
            "student_macro_f1": summary["student_macro_f1"],
            "student_top1": summary["student_top1_when_candidate"],
        })

    df_sweep = pd.DataFrame(cv_scores)
    out_dir = Path("outputs/plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "knn_k_sensitivity.csv"
    df_sweep.to_csv(csv_path, index=False)

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        df_sweep["k"],
        df_sweep["cv_macro_f1"],
        yerr=df_sweep["cv_macro_f1_std"],
        marker="o",
        capsize=5,
        color="darkorange",
        label="CV Macro F1 (mean ± std)",
    )
    plt.plot(df_sweep["k"], df_sweep["student_macro_f1"], marker="s", linestyle="--", color="gray", label="Student-test Macro F1")
    plt.title("KNN Sensitivity: k vs Macro F1")
    plt.xlabel("k (Number of Neighbors)")
    plt.ylabel("Macro F1")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    fig_path = out_dir / "knn_k_sensitivity.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved KNN k-sensitivity plot to {fig_path} and CSV to {csv_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="knn")
    ap.add_argument("--k", type=int, default=21)
    ap.add_argument("--weights", default="distance", choices=["distance", "uniform"])
    ap.add_argument("--metric", default="minkowski")
    ap.add_argument("--subsample-negatives", action="store_true", default=True)
    ap.add_argument("--no-subsample-negatives", action="store_false", dest="subsample_negatives")
    ap.add_argument("--negative-ratio", type=float, default=4.0)
    ap.add_argument("--max-train-samples", type=int, default=50000)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--sweep-k", action="store_true", help="Run sensitivity sweep over k=[11, 21, 31]")
    args = ap.parse_args()

    if args.sweep_k:
        run_k_sweep(args)
    else:
        cfg = build_config(args)
        run_single_knn(args.name, cfg)


if __name__ == "__main__":
    main()
