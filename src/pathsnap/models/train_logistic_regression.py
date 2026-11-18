#!/usr/bin/env python3
"""Train and evaluate the Logistic Regression candidate scorer (Milestone 3, Person 1).

Usage:
    python -m pathsnap.models.train_logistic_regression                      # initial configuration
    python -m pathsnap.models.train_logistic_regression --C 0.1 --name logistic_regression_C0.1
    python -m pathsnap.models.train_logistic_regression --class-weight none --name logistic_regression_unweighted

Model: L2-regularised logistic regression on standardised features (median imputation + missing indicators).
Loss: (class-weighted) binary cross-entropy + (1 / 2C) * ||w||^2, minimised with L-BFGS.
Imbalance: ``class_weight="balanced"`` (weights n / (2 * n_class)), i.e. positives (~10 % of rows) weighted ~4.8x.

Grouped 5-fold CV on train trips, then fit on all train trips and score student_test; everything is logged to
outputs/results/experiments.csv via pathsnap.models.harness.
"""

from __future__ import annotations

import argparse
import json

from sklearn.linear_model import LogisticRegression

from pathsnap.models import harness as H


def build_config(args: argparse.Namespace) -> dict:
    return {
        "estimator": "sklearn.linear_model.LogisticRegression",
        "C": args.C,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": args.max_iter,
        "class_weight": None if args.class_weight == "none" else args.class_weight,
        "loss": "class-weighted binary cross-entropy + L2" if args.class_weight != "none" else "binary cross-entropy + L2 (unweighted)",
        "preprocessing": "median impute + missing indicators + StandardScaler (in Pipeline)",
        "threshold": 0.5,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--class-weight", default="balanced", choices=["balanced", "none"])
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--name", default="logistic_regression")
    args = ap.parse_args()
    cfg = build_config(args)

    def factory():
        est = LogisticRegression(C=cfg["C"], penalty="l2", solver="lbfgs", max_iter=cfg["max_iter"],
                                 class_weight=cfg["class_weight"], random_state=H.SEED)
        return H.make_pipeline(est, scale=True)

    print(f"== {args.name} ==\nconfig: {json.dumps(cfg)}")
    summary = H.run_experiment(factory, name=args.name, family="linear", config=cfg)
    keys = ["macro_f1", "top1_when_candidate", "top1_all_points", "candidate_miss_rate", "route_continuity",
            "log_loss", "precision", "recall", "accuracy"]
    print(f"\n{'metric':22s} {'fold-train':>10s} {'CV mean':>9s} {'CV std':>8s} {'full-train':>10s} {'student':>9s}")
    for k in keys:
        print(f"{k:22s} {summary[f'cv_train_{k}_mean']:10.4f} {summary[f'cv_{k}_mean']:9.4f} {summary[f'cv_{k}_std']:8.4f} "
              f"{summary[f'full_train_{k}']:10.4f} {summary[f'student_{k}']:9.4f}")
    print(f"student confusion matrix [[TN, FP], [FN, TP]]: {summary['student_confusion_matrix']}")
    print(f"timing: CV fit {summary['cv_fit_time_s_mean']:.1f}s/fold, full fit {summary['full_fit_time_s']:.1f}s, "
          f"inference {summary['student_infer_ms_per_1k_rows']:.2f} ms per 1k rows, largest trip {summary['largest_trip']}")


if __name__ == "__main__":
    main()
