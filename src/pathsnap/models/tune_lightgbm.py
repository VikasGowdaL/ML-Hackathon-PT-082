
"""Tune LightGBM using trip-grouped cross-validation."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier

from pathsnap.models import harness as H


def main():
    # Small, controlled tuning sweep
    configurations = [
        {
            "name": "lightgbm_baseline",
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 50,
        },
        {
            "name": "lightgbm_regularized",
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_child_samples": 100,
        },
        {
            "name": "lightgbm_more_trees",
            "n_estimators": 400,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_child_samples": 50,
        },
        {
            "name": "lightgbm_complex",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 50,
        },
    ]

    train = H.load_train_table()
    results = []

    for config in configurations:
        name = config["name"]

        print(f"\n{'=' * 60}")
        print(f"Training configuration: {name}")
        print(json.dumps(config, indent=2))

        def factory(cfg=config):
            model = LGBMClassifier(
                n_estimators=cfg["n_estimators"],
                learning_rate=cfg["learning_rate"],
                num_leaves=cfg["num_leaves"],
                min_child_samples=cfg["min_child_samples"],
                class_weight="balanced",
                random_state=H.SEED,
                n_jobs=4,
                verbosity=-1,
            )
            return H.make_pipeline(model, scale=False)

        folds = H.cross_validate(
            model_factory=factory,
            name=name,
            family="gradient_boosting",
            config=config,
            train=train,
        )

        results.append({
            **config,
            "cv_macro_f1": folds["val_macro_f1"].mean(),
            "cv_top1": folds["val_top1_when_candidate"].mean(),
            "cv_log_loss": folds["val_log_loss"].mean(),
            "cv_route_continuity": folds["val_route_continuity"].mean(),
            "train_macro_f1": folds["train_macro_f1"].mean(),
        })

    df = pd.DataFrame(results)

    # Top-1 is the primary map-matching ranking metric
    df = df.sort_values(
        ["cv_top1", "cv_macro_f1"],
        ascending=[False, False],
    )

    output = Path("outputs/results/lightgbm_tuning.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print("\n=== LIGHTGBM TUNING RESULTS ===")
    print(df.to_string(index=False))

    print("\nBest configuration by CV Top-1:")
    print(df.iloc[0].to_string())
    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    main()
