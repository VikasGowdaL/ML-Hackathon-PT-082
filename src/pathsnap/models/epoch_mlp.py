"""EpochMLP: shallow MLP trained epoch by epoch with class-balanced sampling, early stopping and real loss history.

Lives in its own module so that fitted pipelines saved with joblib can be loaded from anywhere
(``joblib.load("artifacts/models/mlp.joblib")``). Training script: ``pathsnap.models.train_mlp``.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier

__all__ = ["EpochMLP", "weighted_bce"]


def weighted_bce(y: np.ndarray, p: np.ndarray, w_pos: float, w_neg: float) -> float:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    w = np.where(y == 1, w_pos, w_neg)
    return float(np.sum(w * -(y * np.log(p) + (1 - y) * np.log(1 - p))) / np.sum(w))


class EpochMLP(BaseEstimator, ClassifierMixin):
    """MLPClassifier trained epoch by epoch with class-balanced sampling, early stopping and a loss history."""

    def __init__(self, hidden_layer_sizes: Tuple[int, ...] = (64, 32), alpha: float = 1e-4,
                 learning_rate_init: float = 1e-3, batch_size: int = 512, max_epochs: int = 60,
                 patience: int = 5, min_delta: float = 1e-4, val_fraction: float = 0.1,
                 train_loss_sample: int = 200_000, random_state: int = 42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.val_fraction = val_fraction
        self.train_loss_sample = train_loss_sample
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EpochMLP":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(int)
        cut = int(len(X) * (1 - self.val_fraction))
        X_tr, y_tr, X_va, y_va = X[:cut], y[:cut], X[cut:], y[cut:]
        n, n_pos = len(y_tr), int(y_tr.sum())
        w_pos, w_neg = n / (2 * n_pos), n / (2 * (n - n_pos))
        pos_idx, neg_idx = np.flatnonzero(y_tr == 1), np.flatnonzero(y_tr == 0)
        rng = np.random.default_rng(self.random_state)
        # training loss is measured each epoch on a fixed random sample (cost); validation loss on all held-out rows
        mon = np.sort(rng.choice(n, size=min(self.train_loss_sample, n), replace=False))

        self.mlp_ = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, activation="relu", solver="adam",
                                  alpha=self.alpha, batch_size=self.batch_size,
                                  learning_rate_init=self.learning_rate_init, shuffle=True,
                                  random_state=self.random_state)
        self.classes_ = np.array([0, 1])
        self.history_: List[Dict[str, Any]] = []
        best, best_state, best_epoch, wait = np.inf, None, 0, 0
        for epoch in range(1, self.max_epochs + 1):
            t0 = time.perf_counter()
            idx = np.concatenate([pos_idx, rng.choice(neg_idx, size=min(len(pos_idx), len(neg_idx)), replace=False)])
            rng.shuffle(idx)
            self.mlp_.partial_fit(X_tr[idx], y_tr[idx], classes=self.classes_)
            fit_s = time.perf_counter() - t0
            p_tr = self.mlp_.predict_proba(X_tr[mon])[:, 1]
            p_va = self.mlp_.predict_proba(X_va)[:, 1]
            rec = {
                "epoch": epoch,
                "train_loss": weighted_bce(y_tr[mon], p_tr, w_pos, w_neg),
                "val_loss": weighted_bce(y_va, p_va, w_pos, w_neg),
                "train_loss_unweighted": weighted_bce(y_tr[mon], p_tr, 1.0, 1.0),
                "val_loss_unweighted": weighted_bce(y_va, p_va, 1.0, 1.0),
                "sklearn_epoch_loss": float(self.mlp_.loss_),  # mean minibatch loss incl. L2 term, on the balanced sample
                "epoch_fit_s": fit_s,
            }
            self.history_.append(rec)
            if rec["val_loss"] < best - self.min_delta:
                best, best_epoch, wait = rec["val_loss"], epoch, 0
                best_state = (copy.deepcopy(self.mlp_.coefs_), copy.deepcopy(self.mlp_.intercepts_))
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.mlp_.coefs_, self.mlp_.intercepts_ = best_state
        self.best_epoch_ = best_epoch
        self.n_epochs_ = len(self.history_)
        self.split_ = {"train_rows": int(cut), "val_rows": int(len(X) - cut), "w_pos": w_pos, "w_neg": w_neg}
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.mlp_.predict_proba(np.asarray(X, dtype=np.float32))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
