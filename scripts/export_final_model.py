#!/usr/bin/env python3
"""Export the selected candidate-scoring model for the API (artifacts/models/final_model.joblib).

The selected model is LightGBM with the hyperparameters found by Bayesian optimisation
(src/pathsnap/models/bayesian_tune_lightgbm.py -> outputs/results/lightgbm_best_params.json); see
docs/model_selection.md. This script refits that configuration on all 80 training trips (no tuning, no
student-test or evaluator data) and writes:

    artifacts/models/final_model.joblib   fitted sklearn Pipeline (imputation + LGBMClassifier); git-ignored
    artifacts/models/final_model.json     what was fitted: parameters, features, data, CV/student results

Usage:
    python scripts/export_final_model.py
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import joblib
import lightgbm
import sklearn
from lightgbm import LGBMClassifier

from pathsnap.models import harness as H
from pathsnap.utils.config import get_repo_root


def main() -> None:
    root = get_repo_root()
    results = root / "outputs" / "results"
    best = json.loads((results / "lightgbm_best_params.json").read_text(encoding="utf-8"))
    params = best["parameters"]

    train = H.load_train_table()
    X, y = H.X_y(train)
    model = H.make_pipeline(
        LGBMClassifier(**params, class_weight="balanced", random_state=H.SEED, n_jobs=4, verbosity=-1),
        scale=False,
    )
    t0 = time.perf_counter()
    model.fit(X, y)
    fit_s = time.perf_counter() - t0

    out_dir = root / "artifacts" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "final_model.joblib")

    student_eval = results / "lightgbm_bayesian_evaluation.json"
    meta = {
        "name": "final_model",
        "estimator": "lightgbm.LGBMClassifier in harness.make_pipeline(scale=False)",
        "selected_because": "docs/model_selection.md",
        "parameters": params,
        "class_weight": "balanced",
        "random_state": H.SEED,
        "features": H.FEATURES,
        "training_data": {"partition": "train", "trips": int(train[H.GROUP].nunique()), "rows": int(len(train))},
        "bayesian_search": {"best_cv_top1_when_candidate": best.get("best_cv_top1"),
                            "completed_trials": best.get("n_completed_trials"),
                            "trials_file": "outputs/results/lightgbm_bayesian_trials.csv"},
        "student_test_metrics": json.loads(student_eval.read_text(encoding="utf-8")) if student_eval.exists() else None,
        "fit_time_s": round(fit_s, 2),
        "versions": {"lightgbm": lightgbm.__version__, "scikit-learn": sklearn.__version__},
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "final_model.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"fitted on {meta['training_data']['rows']:,} rows / {meta['training_data']['trips']} trips in {fit_s:.1f} s")
    print(f"wrote {out_dir / 'final_model.joblib'} and final_model.json")


if __name__ == "__main__":
    main()
