
"""Train and evaluate LightGBM using the shared PathSnap harness."""

from __future__ import annotations

import argparse
import json

from lightgbm import LGBMClassifier

from pathsnap.models import harness as H


def main():
    parser = argparse.ArgumentParser(
        description="Train LightGBM candidate arc scorer"
    )

    parser.add_argument("--name", default="lightgbm")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--min-child-samples", type=int, default=50)
    parser.add_argument("--n-jobs", type=int, default=4)

    args = parser.parse_args()

    config = {
        "estimator": "lightgbm.LGBMClassifier",
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "min_child_samples": args.min_child_samples,
        "class_weight": "balanced",
        "random_state": H.SEED,
        "n_jobs": args.n_jobs,
        "preprocessing": "median imputation + missing indicators",
        "loss": "binary log loss",
        "threshold": 0.5,
    }

    def factory():
        estimator = LGBMClassifier(
            n_estimators=config["n_estimators"],
            learning_rate=config["learning_rate"],
            num_leaves=config["num_leaves"],
            max_depth=config["max_depth"],
            min_child_samples=config["min_child_samples"],
            class_weight=config["class_weight"],
            random_state=config["random_state"],
            n_jobs=config["n_jobs"],
            verbosity=-1,
        )

        return H.make_pipeline(estimator, scale=False)

    print(f"\nTraining {args.name}")
    print(json.dumps(config, indent=2))

    summary = H.run_experiment(
        factory,
        name=args.name,
        family="gradient_boosting",
        config=config,
        notes="LightGBM baseline with balanced class weights",
    )

    metrics = [
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

    print("\n=== LightGBM Evaluation ===")
    print(
        f"{'Metric':25s} "
        f"{'CV Mean':>10s} "
        f"{'CV Std':>10s} "
        f"{'Student':>10s}"
    )

    for metric in metrics:
        print(
            f"{metric:25s} "
            f"{summary[f'cv_{metric}_mean']:10.4f} "
            f"{summary[f'cv_{metric}_std']:10.4f} "
            f"{summary[f'student_{metric}']:10.4f}"
        )

    print("\nConfusion matrix:")
    print(summary["student_confusion_matrix"])

    print("\nTraining time:")
    print(summary["full_fit_time_s"], "seconds")


if __name__ == "__main__":
    main()
