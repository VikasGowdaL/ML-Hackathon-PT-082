# Shallow MLP (Milestone 3 · Person 4)

Script: `python -m pathsnap.models.train_mlp` · Harness: `src/pathsnap/models/harness.py` ·
Metrics: [`docs/metrics.md`](../metrics.md) · Raw results: `outputs/results/experiments.csv` (model `mlp`),
`outputs/results/folds/mlp_folds.csv`, `outputs/results/student/mlp_per_trip.csv`,
epoch histories `outputs/results/mlp_history/mlp_history.csv` and `mlp_loss_curves.png`.

## Why it suits the problem

* Learns non-linear interactions between features (e.g. "close **and** aligned with the driving direction"), which the
  linear model cannot represent, while still producing a probability (`predict_proba`) for the HMM.
* Small enough (64 → 32 hidden units, ~5 k weights) to train on 1.4 M rows in about 30 s on a CPU.

## Setup

| Item | Value |
|---|---|
| Data | Milestone 2 feature table, 80 train trips, 36 features (`harness.FEATURES`) |
| Preprocessing | `harness.make_pipeline(..., scale=True)`: median imputation + missing indicators → `StandardScaler` (fitted per fold) |
| CV | `GroupKFold(5)` by `trip_id`; seed 42 |
| Implementation | `EpochMLP` (`src/pathsnap/models/epoch_mlp.py`): sklearn `MLPClassifier` trained one epoch at a time with `partial_fit`, so that real per-epoch losses can be recorded |

### Initial hyperparameters and loss

| Parameter | Value |
|---|---|
| Hidden layers | 2: 64 → 32, ReLU; sigmoid output |
| Optimiser | Adam, learning rate 1e-3, batch size 512 |
| Weight decay | L2, `alpha = 1e-4` |
| Loss | binary cross-entropy + `alpha · ‖W‖² / 2` |
| Class imbalance | Balanced training: each epoch uses all positives plus an equal-size random sample of negatives (a fresh sample every epoch). This is equivalent in expectation to `class_weight="balanced"`; sklearn's MLP has no `class_weight`. Reported losses use the matching weights `w_c = n / (2 · n_c)` |
| Early stopping | Last 10 % of the training rows held out (rows are ordered by trip, so these are whole trips, except at most one boundary trip); stop after 5 epochs without a validation-loss improvement (min delta 1e-4); best-epoch weights restored; max 60 epochs |
| Loss history | Validation loss on all held-out rows each epoch; training loss each epoch on a fixed, seeded 200,000-row random sample of the training rows (scoring all ~1.3 M rows would cost ~8 s per epoch). Check on the final model: sample loss 0.15184 vs 0.15076 on all 1,594,813 training rows (+0.7 %, standard error 0.0018). The sample is the same every epoch, so the curve's shape is preserved; early stopping uses only the full validation loss |

## Results

| Metric | Fold-train | CV mean ± std | Full-train | Student-test |
|---|---|---|---|---|
| Macro F1 | 0.8643 | 0.8595 ± 0.0253 | 0.8751 | 0.8833 |
| Precision (positive) | 0.6361 | 0.6305 ± 0.0535 | 0.6666 | 0.6810 |
| Recall (positive) | 0.9572 | 0.9489 ± 0.0050 | 0.9449 | 0.9497 |
| Accuracy | 0.9382 | 0.9351 ± 0.0193 | 0.9449 | 0.9527 |
| Log loss | 0.1520 | 0.1621 ± 0.0395 | 0.1390 | 0.1216 |
| Top-1 (points with the true arc as candidate) | 0.8539 | 0.8434 ± 0.0343 | 0.8505 | 0.8585 |
| Top-1 (all points) | 0.7592 | 0.7618 ± 0.1006 | 0.7553 | 0.8573 |
| Route continuity | 0.9261 | 0.9232 ± 0.0294 | 0.9281 | 0.9258 |

Student-test confusion matrix `[[TN, FP], [FN, TP]]` = `[[172409, 8510], [962, 18171]]`.

### Epoch history (real, from `mlp_history.csv`)

| Fit | Epochs run | Best epoch | Train loss at best | Val loss at best | s / epoch |
|---|---|---|---|---|---|
| fold 0 | 8 | 3 | 0.1417 | 0.1073 | 1.3 |
| fold 1 | 14 | 9 | 0.1322 | 0.1090 | 1.3 |
| fold 2 | 6 | 1 | 0.1661 | 0.1153 | 1.3 |
| fold 3 | 9 | 4 | 0.1490 | 0.0926 | 1.3 |
| fold 4 | 13 | 8 | 0.1289 | 0.0777 | 1.2 |
| full (80 trips) | 7 | 2 | 0.1518 | 0.0978 | 1.6 |

## Over / underfitting evidence

* **Epoch curves:** training loss falls steadily in every fit, while validation loss reaches its minimum after 1–9 epochs
  and then rises in folds 0, 2 and 3 (e.g. fold 3: 0.093 at epoch 4 → 0.118 at epoch 8). Further epochs fit the training
  trips more than they generalise; early stopping keeps the best epoch.
* **Validation loss is lower than training loss** in every fit. The early-stopping holdout is the last ~10 % of rows,
  i.e. a different set of trips; these trips are easier than the average training trip (fold-level CV loss varies from
  0.120 to 0.227 for the same reason). The comparison is meaningful within a curve, not across the two curves.
* **Scores:** fold-train vs CV macro F1 0.864 vs 0.860 and Top-1 0.854 vs 0.843. A small generalisation gap, larger than
  logistic regression (≈ 0), so the MLP fits more of the training data without clear overfitting. Student-test (0.883 / 0.859)
  is in line with CV.

## Comparison with the other logged models (CV, from `model_comparison.csv`)

| Model | CV macro F1 | CV Top-1 | CV log loss |
|---|---|---|---|
| logistic_regression (balanced) | 0.8460 | 0.8404 | 0.1897 |
| mlp | 0.8595 | 0.8434 | 0.1621 |
| lightgbm | 0.8734 | 0.8431 | 0.1412 |

The MLP improves macro F1 and log loss over balanced logistic regression, but sits below LightGBM on both. **Top-1 is almost unchanged** (0.843 vs
0.840). All models agree on the top-ranked arc for most points, so remaining errors seem to come from the information in the
features (e.g. the two directions of a two-way street are geometrically identical), not from model capacity.

## Strengths and limitations seen

* + Route continuity on par with LightGBM (CV 0.9232 vs 0.9220) and above logistic regression (0.9141).
* + Cheap: ~29 s per fold including early stopping, 1.23 ms per 1,000 rows, 0.10 s for the largest trip (00000084, 92,587 rows).
* − Unstable early stopping: the best epoch ranges from 1 to 9 and validation loss is noisy (fold 1), because the holdout is
  only ~6–8 trips. A grouped holdout with more trips, or a learning-rate schedule, would make it steadier.
* − Probabilities are inflated for positives by the balanced training (precision 0.63 at threshold 0.5), as for balanced LR.
* − Same weak folds as the other models: fold 4 (contains trip 00000033) has the lowest macro F1 (0.824) and Top-1 (0.812).

Reproduce: `python -m pathsnap.models.train_mlp` (≈ 4 min including CV). Fitted model: `artifacts/models/mlp.joblib` (git-ignored).
