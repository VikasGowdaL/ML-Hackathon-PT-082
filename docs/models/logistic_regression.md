# Logistic Regression (Milestone 3 · Person 1)

Script: `python -m pathsnap.models.train_logistic_regression` · Harness: `src/pathsnap/models/harness.py` ·
Metric definitions: [`docs/metrics.md`](../metrics.md) · Raw results: `outputs/results/experiments.csv`,
`outputs/results/folds/logistic_regression*_folds.csv`, `outputs/results/student/logistic_regression*_per_trip.csv`.

## Why it suits the problem

* Outputs a probability through `predict_proba` directly, which is what the HMM emission step needs.
* A linear baseline: fast (≈ 23 s per fold on ~1.4 M rows, ≈ 1 ms per 1,000 rows to score), deterministic, and the
  coefficients are easy to inspect. It sets the bar the non-linear models have to beat.

## Setup

| Item | Value |
|---|---|
| Data | Milestone 2 feature table, 80 train trips (1,772,015 rows, 10.45 % positive); 36 features (`harness.FEATURES`) |
| Preprocessing | `Pipeline`: median imputation + missing-value indicators → `StandardScaler` → model (fitted per fold) |
| CV | `GroupKFold(5)` by `trip_id` (16 trips per validation fold, disjointness asserted) |
| Seed | 42 |

### Initial hyperparameters and loss

| Parameter | Initial value |
|---|---|
| `penalty` | `l2` |
| `C` | 1.0 |
| `solver` | `lbfgs`, `max_iter = 1000` (converged in 40 iterations on the full train set) |
| `class_weight` | `balanced`: weight n / (2 · n_class), i.e. ≈ 4.8 for positives and ≈ 0.56 for negatives |
| Decision threshold | 0.5 (for row-level metrics only; Top-1 uses the ranking) |
| Loss | class-weighted binary cross-entropy + (1 / 2C) · ‖w‖² |

**Imbalance handling:** `class_weight="balanced"` (required by the project rules). The unweighted variant was run as a
comparison (below).

## Results

All numbers come from `experiments.csv`. "Fold-train" = mean over folds of the score on the training part of each fold.

### Initial model `logistic_regression` (C = 1, balanced)

| Metric | Fold-train | CV mean ± std | Student-test |
|---|---|---|---|
| Macro F1 | 0.8453 | 0.8460 ± 0.0250 | 0.8575 |
| Precision (positive) | 0.5971 | 0.6015 ± 0.0479 | 0.6182 |
| Recall (positive) | 0.9469 | 0.9445 ± 0.0125 | 0.9523 |
| Accuracy | 0.9276 | 0.9273 ± 0.0206 | 0.9392 |
| Log loss | 0.1883 | 0.1897 ± 0.0411 | 0.1643 |
| Top-1 (points with the true arc as candidate) | 0.8406 | 0.8404 ± 0.0330 | 0.8546 |
| Top-1 (all points) | 0.7475 | 0.7591 ± 0.1003 | 0.8534 |
| Candidate miss rate | 0.1105 | 0.0933 ± 0.1394 | 0.0014 |
| Route continuity | 0.9174 | 0.9141 ± 0.0294 | 0.9196 |

Student-test confusion matrix `[[TN, FP], [FN, TP]]` = `[[169664, 11255], [913, 18220]]`.

### Light manual sweep (selected on CV, never on student-test)

| Run | C | class_weight | CV macro F1 | CV Top-1 (cand.) | CV log loss | CV precision / recall |
|---|---|---|---|---|---|---|
| `logistic_regression` | 1.0 | balanced | 0.8460 ± 0.0250 | 0.8404 | 0.1897 | 0.602 / 0.945 |
| `logistic_regression_C0.1` | 0.1 | balanced | 0.8460 ± 0.0251 | 0.8404 | 0.1897 | 0.602 / 0.945 |
| `logistic_regression_unweighted` | 1.0 | none | 0.9004 ± 0.0194 | 0.8400 | 0.1074 | 0.829 / 0.814 |

* **C has no measurable effect** (identical to 4 decimals). With ~1.4 M rows and 48 inputs, the L2 penalty is negligible.
* **Class weighting changes the threshold, not the ranking.** Top-1 and route continuity are the same with and without
  weights, because one weight per class mainly shifts the intercept, which does not change the order of candidates within an
  observation. Weighting raises recall (0.94 vs 0.81) at the cost of precision (0.60 vs 0.83), lowers row-level macro F1, and
  roughly doubles log loss (0.190 vs 0.107): the balanced model's probabilities are inflated for positives.
* **Selected configuration:** `logistic_regression` (C = 1, balanced), as the project rules require class weights. For the
  HMM, the unweighted model's probabilities are better calibrated (log loss 0.107). Either use the unweighted variant as the
  emission model, or renormalise the balanced scores over each observation's candidates. Team decision.

## Over / underfitting evidence

* **No overfitting:** fold-train and CV scores are almost identical (macro F1 0.8453 vs 0.8460, Top-1 0.8406 vs 0.8404,
  log loss 0.1883 vs 0.1897). The full-train fit (0.8452) agrees with both.
* **Underfitting (model bias):** the training score itself is limited to ~0.84 Top-1. A linear boundary cannot represent
  interactions such as "small distance *and* small bearing difference". Non-linear models (trees, MLP) are expected to do better.
* **Student-test is in line with CV** (macro F1 0.8575 vs 0.846; Top-1 0.855 vs 0.840). It is slightly higher because the 10
  student trips contain almost no candidate misses (0.14 %).

## Strengths and limitations seen

* + Fast, stable, trip-independent: every fold converges; CV std of Top-1 is 0.033.
* + Cheap at inference: 0.95 ms per 1,000 rows; the largest trip (00000084, 92,587 rows) scores in 0.09 s.
* − The variance between folds comes mostly from the data, not the model. Fold 2 contains trip 00000054 (GPS offset from the
  map: candidate miss rate 0.34 in that fold). Fold 4, the weakest fold (macro F1 0.811, Top-1 0.812), contains trip 00000033.
* − Ranking errors remain on hard trips: student trip 00000077 reaches only 0.678 Top-1 (others 0.83–0.95).
* − Balanced weights make the probabilities poorly calibrated (see sweep).

## Cost

| | Value |
|---|---|
| Fit per CV fold (~1.42 M rows) | 22.8 s mean |
| Fit on all 80 train trips | 28.4 s |
| Inference | 0.95 ms per 1,000 rows (student-test) |
| Largest trip (00000084, 92,587 rows) | 0.092 s |

Reproduce: `python -m pathsnap.models.train_logistic_regression` (≈ 3 min including CV). The fitted model is written to
`artifacts/models/logistic_regression.joblib` (git-ignored).
