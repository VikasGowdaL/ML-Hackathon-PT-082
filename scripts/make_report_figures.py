#!/usr/bin/env python3
"""Report figures generated from committed result files (no new experiments).

Usage:
    python scripts/make_report_figures.py

Inputs:  outputs/results/pipeline_student_test.csv, lightgbm_bayesian_trials.csv, model_comparison.csv
Outputs: reports/figures/hmm_vs_ml_per_trip.png, bayesian_trials.png, tuned_vs_untuned.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap.utils.config import get_repo_root


def main() -> None:
    root = get_repo_root()
    res, out = root / "outputs" / "results", root / "reports" / "figures"
    out.mkdir(parents=True, exist_ok=True)

    # 1. ML only vs ML + HMM per student_test trip
    p = pd.read_csv(res / "pipeline_student_test.csv", dtype={"trip_id": str})
    x = np.arange(len(p))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    w = 0.38
    axes[0].bar(x - w / 2, p.ml_top1, w, label="ML only (highest probability)", color="#94a3b8")
    axes[0].bar(x + w / 2, p.hmm_top1, w, label="ML + HMM (Viterbi)", color="#a855f7")
    axes[0].set(xticks=x, ylim=(0.6, 1.0), ylabel="Top-1 over all GPS points", title="Observation-level Top-1, student_test")
    axes[0].set_xticklabels(p.trip_id, rotation=45, ha="right"); axes[0].legend(loc="lower left")
    axes[1].bar(x - w / 2, p.ml_continuity, w, label="ML only", color="#94a3b8")
    axes[1].bar(x + w / 2, p.hmm_continuity, w, label="ML + HMM", color="#a855f7")
    axes[1].set(xticks=x, ylim=(0.8, 1.01), ylabel="share of consecutive arcs connected", title="Route continuity, student_test")
    axes[1].set_xticklabels(p.trip_id, rotation=45, ha="right"); axes[1].legend(loc="lower left")
    plt.tight_layout(); fig.savefig(out / "hmm_vs_ml_per_trip.png", dpi=130); plt.close(fig)

    # 2. Bayesian optimisation history
    t = pd.read_csv(res / "lightgbm_bayesian_trials.csv")
    t = t[t.state == "COMPLETE"].sort_values("number")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t.number, t.value, "o", color="#2563eb", label="trial CV Top-1")
    ax.plot(t.number, t.value.cummax(), "-", color="#a855f7", label="best so far")
    ax.axhline(0.8431, color="#64748b", ls="--", lw=1, label="untuned LightGBM (0.8431)")
    ax.set(xlabel="Optuna trial", ylabel="grouped 5-fold CV Top-1", title="Bayesian optimisation of LightGBM (TPE, 21 completed trials)")
    ax.legend(); plt.tight_layout(); fig.savefig(out / "bayesian_trials.png", dpi=130); plt.close(fig)

    # 3. Untuned vs tuned LightGBM (same harness protocol)
    c = pd.read_csv(res / "model_comparison.csv").set_index("model")
    rows = [("CV macro F1", "cv_macro_f1_mean"), ("CV Top-1", "cv_top1_when_candidate"),
            ("Student Top-1", "student_top1_when_candidate"), ("Student macro F1", "student_macro_f1"),
            ("Student log loss", "student_log_loss")]
    a = [float(c.loc["lightgbm", k]) for _, k in rows]
    b = [float(c.loc["lightgbm_tuned", k]) for _, k in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - w / 2, a, w, label="untuned (lightgbm)", color="#94a3b8")
    ax.bar(x + w / 2, b, w, label="tuned (lightgbm_tuned)", color="#2563eb")
    for i, (u, v) in enumerate(zip(a, b)):
        ax.text(i - w / 2, u + 0.01, f"{u:.4f}", ha="center", fontsize=8)
        ax.text(i + w / 2, v + 0.01, f"{v:.4f}", ha="center", fontsize=8)
    ax.set(xticks=x, ylim=(0, 1.05), title="LightGBM before and after Bayesian optimisation (identical protocol)")
    ax.set_xticklabels([r for r, _ in rows]); ax.legend(loc="center right")
    plt.tight_layout(); fig.savefig(out / "tuned_vs_untuned.png", dpi=130); plt.close(fig)
    print("wrote", ", ".join(p.name for p in sorted(out.glob("*.png"))))


if __name__ == "__main__":
    main()
