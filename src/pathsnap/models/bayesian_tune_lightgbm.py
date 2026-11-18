
"""Bayesian hyperparameter optimization for PathSnap LightGBM."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

from pathsnap.models import harness as H
from pathsnap.utils.config import get_repo_root


def make_model(params):
    estimator = LGBMClassifier(
        **params,
        class_weight="balanced",
        random_state=H.SEED,
        n_jobs=4,
        verbosity=-1,
    )
    return H.make_pipeline(estimator, scale=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()

    root = get_repo_root()
    results_dir = root / "outputs" / "results"
    model_dir = root / "artifacts" / "models"

    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    train = H.load_train_table()
    X, y = H.X_y(train)
    groups = train[H.GROUP].to_numpy()

    folds = list(
        GroupKFold(n_splits=H.N_SPLITS).split(X, y, groups)
    )

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators", 100, 600, step=50
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.15, log=True
            ),
            "num_leaves": trial.suggest_int(
                "num_leaves", 15, 63
            ),
            "max_depth": trial.suggest_int(
                "max_depth", 3, 12
            ),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 10, 100
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", 0.0, 5.0
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 0.0, 5.0
            ),
        }

        scores = []
        macro_f1_scores = []
        log_losses = []
        start = time.perf_counter()

        for tr_idx, va_idx in folds:
            model = make_model(params)
            model.fit(X[tr_idx], y[tr_idx])

            probabilities = model.predict_proba(
                X[va_idx]
            )[:, 1]

            validation = train.iloc[va_idx]

            # Rank candidate road arcs independently
            # for each GPS observation.
            scored = validation[
                ["trip_id", "obs_idx", "arc_id", H.TARGET]
            ].copy()

            scored["probability"] = probabilities

            top = (
                scored.sort_values(
                    ["trip_id", "obs_idx", "probability"],
                    ascending=[True, True, False],
                    kind="mergesort",
                )
                .drop_duplicates(
                    ["trip_id", "obs_idx"], keep="first"
                )
            )

            # Matches the existing harness definition:
            # correct Top-1 / observations where the
            # reference road arc is in the candidate set.
            reference_present = (
                scored.groupby(
                    ["trip_id", "obs_idx"]
                )[H.TARGET].max().sum()
            )

            correct = top[H.TARGET].sum()

            score = (
                float(correct / reference_present)
                if reference_present else 0.0
            )

            scores.append(score)

            from sklearn.metrics import f1_score, log_loss

            predictions = (
                probabilities >= 0.5
            ).astype(int)

            macro_f1_scores.append(
                f1_score(
                    y[va_idx],
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            )

            log_losses.append(
                log_loss(
                    y[va_idx],
                    np.clip(
                        probabilities, 1e-7, 1 - 1e-7
                    ),
                    labels=[0, 1],
                )
            )

        trial.set_user_attr(
            "cv_macro_f1", float(np.mean(macro_f1_scores))
        )
        trial.set_user_attr(
            "cv_log_loss", float(np.mean(log_losses))
        )
        trial.set_user_attr(
            "duration_seconds",
            time.perf_counter() - start,
        )

        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=H.SEED)

    storage = (
        "sqlite:///"
        + str(results_dir / "lightgbm_optuna.db")
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="pathsnap_lightgbm",
        storage=storage,
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=args.trials)

    print("\n=== BEST BAYESIAN CONFIGURATION ===")
    print("Best CV Top-1:", study.best_value)
    print(json.dumps(study.best_params, indent=2))

    trials = study.trials_dataframe()
    trials.to_csv(
        results_dir / "lightgbm_bayesian_trials.csv",
        index=False,
    )

    with open(
        results_dir / "lightgbm_best_params.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "best_cv_top1": study.best_value,
                "parameters": study.best_params,
                "n_completed_trials": len(
                    study.get_trials(
                        states=[
                            optuna.trial.TrialState.COMPLETE
                        ]
                    )
                ),
            },
            file,
            indent=2,
        )

    # Refit exclusively on the training partition.
    best_model = make_model(study.best_params)
    best_model.fit(X, y)

    model_path = model_dir / "lightgbm_bayesian.joblib"
    joblib.dump(best_model, model_path)

    print("\nSaved tuned model:", model_path)

    # Student-test is used only after optimization.
    student = H.load_student_table()
    X_student, _ = H.X_y(student)

    probabilities = best_model.predict_proba(
        X_student
    )[:, 1]

    metrics = H.compute_metrics(
        student,
        probabilities,
        H.observation_counts(),
    )

    report_path = (
        results_dir / "lightgbm_bayesian_evaluation.json"
    )

    with open(
        report_path, "w", encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=2)

    print("\n=== STUDENT-TEST EVALUATION ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
