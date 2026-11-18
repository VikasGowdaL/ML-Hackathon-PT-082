# LightGBM — Milestone 3 Model Training

## 1. Objective

Train a LightGBM classifier to predict the probability that a candidate road arc matches a GPS observation's reference road arc.

LightGBM replaces XGBoost as one of the five candidate-scoring models evaluated in Phase 3.

## 2. Dataset and preprocessing

- Training data: Phase 2 candidate feature table.
- Features: 36 model input features selected by the shared training harness.
- Target: Binary candidate label (1 = correct reference arc, 0 = incorrect arc).
- Missing values: Median imputation with missing-value indicators.
- Feature scaling: Not required for gradient-boosted trees.
- Class imbalance: `class_weight="balanced"`.
- Random seed: 42.

## 3. Initial hyperparameters

| Parameter | Value |
|---|---|
| Number of estimators | 200 |
| Learning rate | 0.05 |
| Number of leaves | 31 |
| Maximum depth | -1 |
| Minimum child samples | 50 |
| Class weights | Balanced |

The model uses binary log loss as its training objective.

## 4. Training and evaluation

The model was evaluated using five-fold GroupKFold cross-validation grouped by trip ID. Training and validation trips were kept separate.

The final baseline model was trained on all training trips and evaluated on the student-test partition.

## 5. Baseline results

| Metric | CV mean | CV std | Student-test |
|---|---:|---:|---:|
| Macro F1 | 0.8734 | 0.0234 | 0.8876 |
| Top-1 accuracy (when correct candidate exists) | 0.8431 | 0.0346 | 0.8598 |
| Top-1 accuracy (all observations) | 0.7614 | 0.0996 | 0.8586 |
| Log loss | 0.1412 | 0.0381 | 0.1163 |
| Route continuity | 0.9220 | 0.0315 | 0.9243 |
| Precision | 0.6673 | 0.0506 | 0.6893 |
| Recall | 0.9389 | 0.0132 | 0.9557 |
| Accuracy | 0.9434 | 0.0173 | 0.9546 |

Full-data training time: approximately 9.22 seconds.

## 6. Comparison with Logistic Regression

| Metric | Logistic Regression | LightGBM |
|---|---:|---:|
| CV Macro F1 | 0.8460 | 0.8734 |
| CV Top-1 accuracy | 0.8404 | 0.8431 |
| CV log loss | 0.1897 | 0.1412 |
| CV route continuity | 0.9141 | 0.9220 |

LightGBM improved Macro F1, log loss, and route continuity compared with the balanced Logistic Regression baseline. The improvement in Top-1 accuracy was small.

## 7. Exploratory hyperparameter tuning

Four configurations were tested using grouped cross-validation.

| Configuration | CV Macro F1 | CV Top-1 | CV log loss |
|---|---:|---:|---:|
| Baseline | 0.8734 | 0.8431 | 0.1412 |
| Regularized | 0.8693 | 0.8421 | 0.1475 |
| More trees | 0.8748 | 0.8429 | 0.1393 |
| Complex | 0.8826 | 0.8435 | 0.1287 |

The complex configuration performed best in this limited tuning sweep, although its Top-1 improvement was marginal and its training-validation Macro F1 gap increased.

These experiments are exploratory. The baseline configuration is retained for the initial five-model comparison.

## 8. Limitations

- Candidate generation misses cannot be corrected by the classifier.
- Performance varies across validation folds and GPS trips.
- Class weighting may affect probability calibration.
- Row-level candidate classification does not guarantee globally consistent routes.
- Final model selection should consider all five models under the same evaluation protocol.

## 9. Reproduction

Baseline training:

```bash
PYTHONPATH=src python -m pathsnap.models.train_lightgbm
```

Exploratory tuning:

```bash
PYTHONPATH=src python -m pathsnap.models.tune_lightgbm
```

The shared harness saves experiment metrics, fold-level evaluation, student-test per-trip results, and the fitted baseline model.