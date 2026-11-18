#!/usr/bin/env python3
"""Train and evaluate a shallow MLP candidate scorer (Milestone 3, Person 4).

Usage:
    python -m pathsnap.models.train_mlp                                   # initial configuration
    python -m pathsnap.models.train_mlp --hidden 32 --name mlp_h32        # one hidden layer of 32 units

Model: ``sklearn.neural_network.MLPClassifier`` trained epoch by epoch (``partial_fit``) inside
:class:`EpochMLP`, so that the *real* training and validation loss of every epoch can be recorded.

* Inputs: standardised (``harness.make_pipeline(..., scale=True)``: median impute + indicators + StandardScaler).
* Architecture: ReLU hidden layers (default 64 -> 32), sigmoid output.
* Loss: binary cross-entropy + L2 weight decay ``alpha * ||W||^2 / 2`` (Adam optimiser).
* Imbalance: class-balanced training. Each epoch uses every positive row and an equal-size random sample of
  negatives (a new sample every epoch), which is equivalent in expectation to ``class_weight="balanced"``.
  Reported losses use the same balanced weights (``w_c = n / (2 * n_c)``).
* Early stopping: the last ``val_fraction`` of the training rows is held out. Rows are stored trip by trip, so this
  holdout consists of whole trips (except at most one trip cut at the boundary). Training stops after ``patience``
  epochs without a validation-loss improvement and the best epoch's weights are restored.
* Loss history: validation loss on all held-out rows every epoch; training loss every epoch on a fixed, seeded
  random sample of 200,000 training rows (scoring all ~1.3 M rows each epoch would cost ~8 s per epoch).

Per-epoch histories are saved to ``outputs/results/mlp_history/<name>_*.csv`` and plotted.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, List

import numpy as np
import pandas as pd
from pathsnap.models import harness as H
from pathsnap.models.epoch_mlp import EpochMLP, weighted_bce  # noqa: F401  (importable class -> loadable joblib)


def save_histories(models: List[Any], name: str) -> pd.DataFrame:
    """Write per-epoch histories of the CV folds (0-4) and the final fit ('full') and plot them."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = H._paths()["results"] / "mlp_history"
    out.mkdir(parents=True, exist_ok=True)
    labels = [f"fold{k}" for k in range(len(models) - 1)] + ["full"]
    frames = []
    for lab, m in zip(labels, models):
        est = m.named_steps["model"]
        h = pd.DataFrame(est.history_).assign(run=lab, best_epoch=est.best_epoch_)
        frames.append(h)
    hist = pd.concat(frames, ignore_index=True)
    hist.to_csv(out / f"{name}_history.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for ax, (lab, h) in zip(axes.ravel(), hist.groupby("run", sort=False)):
        ax.plot(h.epoch, h.train_loss, marker="o", ms=3, label="train (balanced BCE)")
        ax.plot(h.epoch, h.val_loss, marker="o", ms=3, label="validation (balanced BCE)")
        ax.axvline(h.best_epoch.iloc[0], color="0.5", ls="--", lw=1, label="best epoch")
        ax.set(title=lab, xlabel="epoch", ylabel="loss")
        ax.legend(fontsize=8)
    fig.suptitle(f"{name}: real per-epoch training and validation loss")
    plt.tight_layout()
    fig.savefig(out / f"{name}_loss_curves.png", dpi=110)
    plt.close(fig)
    return hist


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 32])
    ap.add_argument("--alpha", type=float, default=1e-4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--name", default="mlp")
    args = ap.parse_args()
    cfg = {
        "estimator": "EpochMLP(sklearn MLPClassifier, partial_fit per epoch)",
        "hidden_layer_sizes": list(args.hidden),
        "activation": "relu",
        "solver": "adam",
        "learning_rate_init": args.lr,
        "alpha_l2_weight_decay": args.alpha,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "early_stopping": f"val loss, patience {args.patience}, min_delta 1e-4, restore best",
        "val_fraction": 0.1,
        "train_loss_monitor_rows": 200000,
        "class_weight": "balanced (all positives + equal-size fresh negative sample per epoch)",
        "loss": "binary cross-entropy + L2 weight decay",
        "preprocessing": "median impute + missing indicators + StandardScaler (in Pipeline)",
        "threshold": 0.5,
    }
    created: List[Any] = []

    def factory():
        m = H.make_pipeline(EpochMLP(hidden_layer_sizes=tuple(args.hidden), alpha=args.alpha,
                                     learning_rate_init=args.lr, batch_size=args.batch_size,
                                     max_epochs=args.max_epochs, patience=args.patience,
                                     random_state=H.SEED), scale=True)
        created.append(m)
        return m

    print(f"== {args.name} ==\nconfig: {json.dumps(cfg)}")
    summary = H.run_experiment(factory, name=args.name, family="neural_network", config=cfg)
    hist = save_histories(created, args.name)
    print("\nepochs run / best epoch per fit:")
    for run, h in hist.groupby("run", sort=False):
        b = h[h.epoch == h.best_epoch]
        print(f"  {run:6s}: {len(h):2d} epochs, best {int(h.best_epoch.iloc[0]):2d} | train {b.train_loss.iloc[0]:.4f} "
              f"val {b.val_loss.iloc[0]:.4f} | {h.epoch_fit_s.mean():.1f}s/epoch")
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
