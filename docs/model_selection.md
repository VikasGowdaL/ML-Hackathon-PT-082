# Model Selection (Milestone 3)

Source data: `outputs/results/experiments.csv`, `outputs/results/model_comparison.csv`,
`outputs/results/folds/*_folds.csv`, `outputs/results/lightgbm_bayesian_trials.csv`. Metric definitions: [`metrics.md`](metrics.md).

## Criteria (fixed before tuning)

The candidate scores feed the HMM / Viterbi decoder, which needs the correct arc ranked first at each GPS point and
probabilities it can use as emission scores. Ranking, in order:

| # | Criterion | Why |
|---|---|---|
| 1 | **CV Top-1 (points whose true arc is a candidate)** | Threshold-free; it is what Viterbi consumes; not distorted by class weights |
| 2 | CV route continuity | Consecutive top-1 arcs should form a drivable path |
| 3 | CV log loss | Probability quality for emissions (only comparable between models trained the same way) |
| 4 | Fold stability (std), inference time, fit time | Live API latency; feasibility of Bayesian optimisation |
| — | **Not used:** training scores, accuracy (≈ 90 % negatives), macro F1 at threshold 0.5 (depends on class weights), student-test results | Student-test is a final check only |

## Evidence (grouped 5-fold CV, identical trip splits for every model)

| Model | CV Top-1 | Route continuity | Log loss | CV macro F1 | Fit (all train) | Inference / 1k rows |
|---|---|---|---|---|---|---|
| random_forest | **0.8461** | **0.9245** | 0.1354 | 0.8799 | 201.5 s | 2.38 ms |
| mlp | 0.8434 | 0.9232 | 0.1621 | 0.8595 | 29.6 s | 1.23 ms |
| lightgbm | 0.8431 | 0.9220 | 0.1412 | 0.8734 | 9.2 s | 1.74 ms |
| logistic_regression | 0.8404 | 0.9141 | 0.1897 | 0.8460 | 28.4 s | 0.95 ms |
| logistic_regression_unweighted | 0.8400 | 0.9151 | 0.1074 | 0.9004 | 26.1 s | 1.11 ms |
| knn | 0.8228 | 0.8977 | 0.1624 | 0.8763 | 15.4 s | 61.35 ms |

Paired fold-by-fold comparison of Top-1 (same 16 validation trips per fold for every model):

* Random Forest vs LightGBM: +0.0030 on average, better in 3 / 5 folds; the lead comes almost entirely from fold 3
  (0.8594 vs 0.8439). In the other four folds the difference is within ±0.002.
* Random Forest vs MLP: +0.0027, better in only 2 / 5 folds.
* Random Forest, LightGBM and MLP each beat logistic regression and KNN on Top-1 and route continuity in 5 / 5 folds
  (RF shown: +0.006 vs LR, +0.023 vs KNN).

**Conclusion:** Random Forest, LightGBM and the MLP are statistically tied on the primary criterion; all three are
clearly better than logistic regression and KNN.

## Decision

**LightGBM** is selected for Bayesian optimisation and as the final model:

* tied for best Top-1 / continuity, second-best log loss among the balanced models (0.141 vs RF 0.135, MLP 0.162);
* fits in ~9 s versus ~200 s for Random Forest, so a Bayesian search with 5-fold CV per trial is feasible:
  a LightGBM trial took ~1 min (measured: 58.7 s for trial 0), whereas Random Forest needs ~4 min per fold
  (measured 224–243 s), i.e. ~20 min per trial; 50 trials ≈ 50 min vs ≈ 17 h;
* fast inference (1.7 ms per 1,000 rows) for the live API.

## Bayesian optimisation (Optuna TPE, LightGBM only)

`src/pathsnap/models/bayesian_tune_lightgbm.py` — objective: mean grouped-CV Top-1 (when the true arc is a candidate),
same folds and seed (42); 22 trials run, 21 completed (1 failed).

| | Value |
|---|---|
| Trial CV Top-1 range | 0.8408 – 0.8454 (median 0.8434) |
| Best CV Top-1 | **0.8454** (baseline LightGBM 0.8431) |
| Best parameters | `n_estimators=200, learning_rate=0.0260, num_leaves=42, max_depth=9, min_child_samples=50, colsample_bytree=0.754, reg_alpha=4.81, reg_lambda=0.0229`, `class_weight="balanced"` |
| Student-test (after tuning, checked once) | Top-1 0.8574, macro F1 0.8834, log loss 0.1233, route continuity 0.9234 |

The gain from tuning is small (+0.002 Top-1), consistent with all models sitting near 0.84: the remaining errors are
limited by the features (e.g. the two directions of a two-way street are geometrically identical), and are largely
recovered by the HMM (below).

## Final model and pipeline

* Exported by `python scripts/export_final_model.py` → `artifacts/models/final_model.joblib` (tracked, 0.96 MB) and
  `final_model.json` (parameters, features, data, results). Refit on all 80 train trips.
* Used by `src/pathsnap/pipeline.py` with Viterbi decoding and shortest-path route reconstruction.

Student-test, full pipeline (`python scripts/evaluate_pipeline.py`, `outputs/results/pipeline_student_test*.{csv,json}`):

| | ML only (highest probability per point) | ML + HMM (Viterbi) |
|---|---|---|
| Top-1 over all 19,160 GPS points | 0.856 | **0.876** |
| Consecutive predicted arcs connected | 0.924 | **0.996** |
| Route F1 / precision / recall (by length, mean of 10 trips) | — | **0.933** / 0.886 / 0.990 |
| Prediction time per trip | — | 0.85 s mean, 1.9 s max |
