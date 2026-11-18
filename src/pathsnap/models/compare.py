#!/usr/bin/env python3
"""Compare all logged candidate-scoring models (Milestone 3, Person 4).

Usage:
    python -m pathsnap.models.compare

Reads ``outputs/results/experiments.csv`` (written by ``pathsnap.models.harness``), keeps the latest run per model
name, and writes:

* ``outputs/results/model_comparison.csv`` -- one row per model: model, family, key config, CV macro F1 (mean, std),
  CV Top-1, route continuity, student-test macro F1 / Top-1, log loss, inference time. Metrics that do not apply or
  were not logged are ``N/A`` with the reason in ``na_reason``.
* ``outputs/results/figures/``: ``cv_macro_f1.png`` (mean +/- std over folds), ``student_top1_macro_f1.png``,
  ``log_loss.png``, ``confusion_matrices.png`` (student-test, row-normalised).

No metric is computed here; everything is read from the experiment log.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from pathsnap.utils.config import get_repo_root

# config keys that describe the run's setup rather than hyperparameters
_CONFIG_SKIP = {"estimator", "loss", "preprocessing", "features", "n_features", "cv", "threshold", "random_state",
                "n_jobs", "early_stopping", "val_fraction", "train_loss_monitor_rows", "activation", "solver", "penalty"}

COLUMNS = [
    ("cv_macro_f1_mean", "cv_macro_f1_mean"),
    ("cv_macro_f1_std", "cv_macro_f1_std"),
    ("cv_top1_when_candidate", "cv_top1_when_candidate_mean"),
    ("cv_top1_all_points", "cv_top1_all_points_mean"),
    ("cv_route_continuity", "cv_route_continuity_mean"),
    ("cv_log_loss", "cv_log_loss_mean"),
    ("cv_train_macro_f1", "cv_train_macro_f1_mean"),
    ("student_macro_f1", "student_macro_f1"),
    ("student_top1_when_candidate", "student_top1_when_candidate"),
    ("student_top1_all_points", "student_top1_all_points"),
    ("student_route_continuity", "student_route_continuity"),
    ("student_log_loss", "student_log_loss"),
    ("full_fit_time_s", "full_fit_time_s"),
    ("infer_ms_per_1k_rows", "student_infer_ms_per_1k_rows"),
]


def key_config(config: Dict[str, Any]) -> str:
    """Short 'k=v; ...' string of the hyperparameters in a logged config."""
    items = [(k, v) for k, v in sorted(config.items()) if k not in _CONFIG_SKIP]
    return "; ".join(f"{k}={json.dumps(v) if isinstance(v, (list, dict)) else v}" for k, v in items)


def load_runs(csv_path: Path) -> pd.DataFrame:
    """Latest run per model name from the experiment log, with parsed config / metrics."""
    log = pd.read_csv(csv_path)
    log["timestamp"] = pd.to_datetime(log["timestamp"], utc=True, errors="coerce")
    log = log.sort_values("timestamp").drop_duplicates("model", keep="last")
    log["config"] = log["config_json"].apply(lambda s: json.loads(s) if isinstance(s, str) and s.strip() else {})
    log["metrics"] = log["metrics_json"].apply(lambda s: json.loads(s) if isinstance(s, str) and s.strip() else {})
    return log.reset_index(drop=True)


def build_comparison(runs: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for r in runs.itertuples():
        m = r.metrics
        row: Dict[str, Any] = {"model": r.model, "family": r.family, "key_config": key_config(r.config),
                               "seed": r.seed, "logged_at": r.timestamp.isoformat() if pd.notna(r.timestamp) else ""}
        missing = []
        for out, src in COLUMNS:
            v = m.get(src)
            row[out] = float(v) if isinstance(v, (int, float)) and np.isfinite(v) else "N/A"
            if row[out] == "N/A":
                missing.append(out)
        if "cv_macro_f1_mean" not in m:
            reason = "run was not produced by the Milestone 3 harness (no grouped-CV / student-test metrics logged)"
        elif missing:
            reason = "not logged: " + ", ".join(missing)
        else:
            reason = ""
        row["na_reason"] = reason
        cm = m.get("student_confusion_matrix")
        row["student_confusion_matrix"] = json.dumps(cm) if cm else "N/A"
        rows.append(row)
    df = pd.DataFrame(rows)
    harness = df[df.na_reason == ""].copy()
    order = harness.sort_values("cv_macro_f1_mean", ascending=False).model.tolist()
    rest = df[df.na_reason != ""].model.tolist()
    return df.set_index("model").loc[order + rest].reset_index()


def make_plots(comp: pd.DataFrame, out_dir: Path) -> List[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    ok = comp[comp.na_reason == ""].copy()
    if ok.empty:
        return []
    for c in [c for c, _ in COLUMNS]:
        ok[c] = pd.to_numeric(ok[c], errors="coerce")
    x = np.arange(len(ok))
    paths = []

    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(ok)), 4.5))
    ax.bar(x, ok.cv_macro_f1_mean, yerr=ok.cv_macro_f1_std, capsize=5, color="#4C72B0")
    ax.scatter(x, ok.cv_train_macro_f1, color="#C44E52", zorder=3, label="fold-train mean")
    ax.set_xticks(x, ok.model, rotation=30, ha="right")
    ax.set(ylabel="macro F1", title="Grouped 5-fold CV macro F1 (mean ± std over folds)",
           ylim=(max(0, ok.cv_macro_f1_mean.min() - 0.1), 1.0))
    ax.legend(); plt.tight_layout(); p = out_dir / "cv_macro_f1.png"; fig.savefig(p, dpi=120); plt.close(fig); paths.append(p)

    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(ok)), 4.5))
    w = 0.38
    ax.bar(x - w / 2, ok.student_top1_when_candidate, w, label="Top-1 (true arc among candidates)", color="#55A868")
    ax.bar(x + w / 2, ok.student_macro_f1, w, label="macro F1", color="#8172B3")
    ax.set_xticks(x, ok.model, rotation=30, ha="right")
    lo = min(ok.student_top1_when_candidate.min(), ok.student_macro_f1.min())
    ax.set(ylabel="score", title="Student-test (10 trips): Top-1 and macro F1", ylim=(max(0, lo - 0.1), 1.0))
    ax.legend(); plt.tight_layout(); p = out_dir / "student_top1_macro_f1.png"; fig.savefig(p, dpi=120); plt.close(fig); paths.append(p)

    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(ok)), 4.5))
    ax.bar(x - w / 2, ok.cv_log_loss, w, label="CV mean", color="#4C72B0")
    ax.bar(x + w / 2, ok.student_log_loss, w, label="student-test", color="#DD8452")
    ax.set_xticks(x, ok.model, rotation=30, ha="right")
    ax.set(ylabel="log loss (lower is better)", title="Probability quality: log loss")
    ax.legend(); plt.tight_layout(); p = out_dir / "log_loss.png"; fig.savefig(p, dpi=120); plt.close(fig); paths.append(p)

    cms = [(r.model, np.array(json.loads(r.student_confusion_matrix))) for r in ok.itertuples()
           if r.student_confusion_matrix != "N/A"]
    if cms:
        n = len(cms); cols = min(4, n); rows_ = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows_, cols, figsize=(3.6 * cols, 3.4 * rows_), squeeze=False)
        for ax, (name, cm) in zip(axes.ravel(), cms):
            norm = cm / cm.sum(axis=1, keepdims=True)
            ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{norm[i, j]:.3f}\n({cm[i, j]:,})", ha="center", va="center", fontsize=8,
                            color="white" if norm[i, j] > 0.5 else "black")
            ax.set_xticks([0, 1], ["pred 0", "pred 1"]); ax.set_yticks([0, 1], ["true 0", "true 1"]); ax.set_title(name, fontsize=9)
        for ax in axes.ravel()[n:]:
            ax.axis("off")
        fig.suptitle("Student-test confusion matrices (row-normalised, threshold 0.5)")
        plt.tight_layout(); p = out_dir / "confusion_matrices.png"; fig.savefig(p, dpi=120); plt.close(fig); paths.append(p)
    return paths


def main() -> None:
    root = get_repo_root()
    results = root / "outputs" / "results"
    runs = load_runs(results / "experiments.csv")
    comp = build_comparison(runs)
    comp.to_csv(results / "model_comparison.csv", index=False)
    paths = make_plots(comp, results / "figures")
    show = ["model", "family", "cv_macro_f1_mean", "cv_macro_f1_std", "cv_top1_when_candidate", "cv_route_continuity",
            "student_macro_f1", "student_top1_when_candidate", "student_log_loss", "infer_ms_per_1k_rows", "na_reason"]
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 60):
        print(comp[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nwrote {results / 'model_comparison.csv'}")
    for p in paths:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
