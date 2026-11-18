# Model Evaluation Metrics (Milestone 3)

Implemented in `src/pathsnap/models/harness.py` (`compute_metrics`). Every model is scored with the same code.

## Evaluation protocol

| Item | Rule |
|---|---|
| Training data | `train` partition only (80 trips), Milestone 2 feature table |
| Cross-validation | `GroupKFold(n_splits=5)` grouped by `trip_id`; trip disjointness asserted per fold |
| Preprocessing | sklearn `Pipeline`: median imputation + missing-value indicators, plus `StandardScaler` for LR / KNN / MLP; fitted inside each fold on the training part only |
| Development test | Fit on all 80 train trips, score `student_test` (10 trips). Not used for model selection or tuning |
| Evaluator data | Never read |
| Seed | 42 everywhere |
| Inputs | 36 features (`harness.FEATURES`); excluded columns and reasons in `harness.EXCLUDED_FEATURES` |

Student-test labels come from Person 2's monotonic reference alignment (`student_test_labels()`), used for evaluation only.

## Row-level metrics (one row = one (GPS point, candidate arc) pair)

Predicted class = 1 if `predict_proba ≥ 0.5`.

| Metric | Definition |
|---|---|
| `precision` | TP / (TP + FP) for the positive class (candidate is the reference arc) |
| `recall` | TP / (TP + FN) for the positive class |
| `macro_f1` | Unweighted mean of the F1 scores of class 0 and class 1 |
| `accuracy` | (TP + TN) / rows. Dominated by the ~90 % negative class; reported but not used for selection |
| `log_loss` | Binary cross-entropy of `predict_proba` (clipped to [1e-7, 1 − 1e-7]). Measures probability quality |
| `confusion_matrix` | `[[TN, FP], [FN, TP]]` at threshold 0.5 |

## Observation-level metrics (one GPS point = all its candidates)

For each GPS point, the **top-1 arc** is the candidate with the highest predicted probability.

| Metric | Definition |
|---|---|
| `candidate_miss_rate` | Share of GPS points whose reference arc is **not** in the candidate set (including points with no candidate). No model can score these points correctly; this depends only on candidate generation |
| `top1_all_points` | Points whose top-1 arc equals the reference arc ÷ **all** GPS points of the evaluated trips (missed points count as wrong) |
| `top1_when_candidate` | Same numerator ÷ points whose reference arc **is** a candidate. Isolates the model's ranking quality. `top1_all_points = top1_when_candidate × (1 − candidate_miss_rate)` |
| `route_continuity` | Over consecutive GPS points of the same trip **and segment** (`segment_id`, split at gaps > 60 s) that have candidates: share where the two top-1 arcs are identical or connected head-to-tail (`to_node(a) == from_node(b)`) in the trip's road graph. 1.0 = a drivable sequence |

## Cost

| Metric | Definition |
|---|---|
| `fit_time_s` | Wall-clock time of `Pipeline.fit` (per fold, and on all 80 train trips) |
| `infer_ms_per_1k_rows` | `predict_proba` wall-clock time per 1,000 rows (validation fold, student_test) |
| `largest_trip` | `predict_proba` time on the trip with the most candidate rows |

## Where results go

| File | Content |
|---|---|
| `outputs/results/experiments.csv` | One row per run: model, family, JSON config, seed, metrics JSON (`cv_<m>_mean`, `cv_<m>_std`, `cv_train_<m>_mean`, `full_train_<m>`, `student_<m>`, timings), train time, inference time (s per 1,000 rows) |
| `outputs/results/folds/<name>_folds.csv` | Per fold: train and validation metrics, rows, timing, confusion matrix |
| `outputs/results/student/<name>_per_trip.csv` | Student-test metrics per trip |
| `artifacts/models/<name>.joblib` | Model fitted on all train trips (git-ignored; regenerate with the training script) |

## Over / underfitting evidence

Compare `cv_train_<m>_mean` (training part of each fold), `cv_<m>_mean` (validation folds) and `student_<m>`. A large
train − CV gap indicates overfitting; low values everywhere indicate underfitting.

## Route reconstruction metrics (full pipeline, API)

Computed by `pathsnap.pipeline.score_prediction` **after** prediction; the prediction itself never reads the route.

| Metric | Definition |
|---|---|
| `route_precision` | Length of predicted-route arcs that are on the reference route ÷ length of the predicted route (unique arcs) |
| `route_recall` | Length of reference-route arcs that were predicted ÷ length of the reference route (unique arcs) |
| `route_f1` | Harmonic mean of the two; shown in the UI as "Route accuracy" |
| `arc_precision` / `arc_recall` | The same, counted in arcs instead of metres |
| `points_on_route` | Share of **all** GPS points whose Viterbi arc belongs to the reference route (no prediction = wrong) |
| `points_on_route_ml_only` | The same for the model's highest-probability arc per point, without the HMM |
| `route_sections` / `unbridged_breaks` | Number of disconnected pieces of the predicted route / consecutive arcs that could not be joined by a road path ≤ 2 km |

`scripts/evaluate_pipeline.py` additionally reports observation-level Top-1 for ML only and ML + HMM against Person 2's
aligned reference arc per point (train / student_test only).
