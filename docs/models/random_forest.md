# Model Specification: Random Forest Classifier

**Target Task**: Score candidate road arcs for GPS observations $(p(\text{label}=1 \mid \mathbf{x}))$.  
**Pipeline Location**: `src/pathsnap/models/train_random_forest.py`  
**Entry Point**: `model_factory(config=None)`, `run(trips=None)`

---

## 1. Rationale: Why Random Forest Suits Candidate Arc Scoring

In map matching, determining whether a road arc matches a GPS point involves non-linear interactions across spatial proximity, angular alignment, and local topology:
- **Non-Linear Interactions**: A candidate arc 15 metres away might be valid if the vehicle heading closely aligns with the arc bearing ($\Delta\theta < 20^\circ$), but invalid if perpendicular ($\Delta\theta \approx 90^\circ$). Decision trees capture these orthogonal threshold boundaries naturally without explicit feature interaction terms.
- **Robustness to Differing Feature Scales**: The feature vector contains a mixture of distances in metres ($[0, 50]$), normalized projection fractions ($[0, 1]$), discrete node degrees ($\ge 0$), and angles in degrees ($[0, 180]$). Random Forest is scale-invariant and requires no normalizer or feature standardization.
- **Noise Resistance**: GPS noise exhibits sporadic jumps, multipath reflections, and drift. Averaging across an ensemble of 200 de-correlated trees mitigates single-tree variance and stabilizes predicted probabilities.

---

## 2. Expected Strengths and Limitations

### Strengths
- **Native Imbalance Handling**: With an 8.57:1 negative-to-positive class imbalance, `class_weight="balanced_subsample"` dynamically weights each bootstrap sample inversely proportional to class frequencies, penalizing missed ground-truth arcs.
- **Probability Smoothness**: Ensembling 200 trees produces fine-grained probability resolution (steps of $\approx 0.005$) suitable for subsequent Viterbi or HMM emission scoring.
- **Fast Inference**: Decision path lookups on depth-15 trees execute in ~14 ms per 1,000 candidate rows.
- **Feature Importance**: Provides Gini impurity and permutation importance to assess the discriminative power of candidate distance vs. angular alignment vs. topological degree.

### Limitations
- **Sequence Blindness**: Random Forest scores each observation-candidate pair independently as an i.i.d. instance. It cannot evaluate topological transition feasibility between consecutive points (left to the downstream HMM/sequence model).
- **Extrapolation**: Cannot extrapolate beyond observed feature bounds (though bounded spatial radius and metric coordinates make this less critical).
- **Model Size on Disk**: An ensemble of 200 depth-15 trees trained on 1.77 million rows can exceed several hundred megabytes if unconstrained.

---

## 3. Initial Hyperparameters & Justifications

All hyperparameters are centralized in `RF_CONFIG` in `src/pathsnap/models/train_random_forest.py`:

| Parameter | Initial Value | Justification |
|---|---|---|
| `n_estimators` | `200` | Sufficient ensemble size to achieve variance reduction and smooth probabilities without excessive memory or training time. |
| `max_depth` | `15` | Constrains tree depth to limit overfitting to noisy individual GPS coordinates; prevents oversized tree leaves. |
| `min_samples_leaf` | `10` | Guards against single-observation leaf nodes, smoothing local probability estimates and mitigating label noise. |
| `class_weight` | `"balanced_subsample"` | Adjusts weights per bootstrap tree inversely proportional to class frequencies: $w_c = \frac{N}{2 \cdot N_c}$, counteracting the 8.57:1 negative-to-positive ratio. |
| `n_jobs` | `-1` | Parallelizes tree construction across all available CPU cores. |
| `random_state` | `42` | Guarantees deterministic splits and reproducible results. |
| `impute_missing` | `True` | Employs `SimpleImputer(strategy="median")` inside the pipeline to resolve the 0.10% NaNs arising from stationary GPS points with undefined heading. |

---

## 4. Objective & Loss

- **Split Criterion**: Gini Impurity:
  $$I_G(p) = 1 - \sum_{k=0}^1 p_k^2$$
- **Scoring Output**: Probability estimate of the positive class via `predict_proba`:
  $$\hat{p}(\text{label}=1 \mid \mathbf{x}) = \frac{1}{B} \sum_{b=1}^B \hat{p}_b(\text{label}=1 \mid \mathbf{x})$$
- **Evaluation Metric (via Harness)**:
  - Binary Cross-Entropy (Log-Loss)
  - Precision-Recall AUC (PR-AUC)
  - Top-1 candidate ranking accuracy per GPS observation

---

## 5. Experimental Results

Run: `python -m pathsnap.models.train_random_forest` (n_estimators 200, max_depth 15, min_samples_leaf 10,
max_features "sqrt", class_weight "balanced_subsample"). Source: `outputs/results/experiments.csv` (model `random_forest`),
`outputs/results/folds/random_forest_folds.csv`, `outputs/results/student/random_forest_per_trip.csv`.
PR-AUC and ROC-AUC are not computed by the shared harness and are therefore not reported.

### Grouped 5-fold CV (validation folds, 16 trips each)

| Metric | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean ± Std |
|---|---|---|---|---|---|---|
| Macro F1 | 0.8746 | 0.8725 | 0.9116 | 0.8942 | 0.8464 | 0.8799 ± 0.0246 |
| Log loss | 0.1442 | 0.1291 | 0.1045 | 0.1115 | 0.1880 | 0.1354 ± 0.0332 |
| Top-1 (true arc among candidates) | 0.8448 | 0.8158 | 0.8969 | 0.8594 | 0.8136 | 0.8461 ± 0.0344 |
| Route continuity | 0.9446 | 0.9189 | 0.9285 | 0.8817 | 0.9486 | 0.9245 ± 0.0268 |
| Precision | 0.6724 | 0.6682 | 0.7571 | 0.7158 | 0.6103 | 0.6848 ± 0.0552 |
| Recall | 0.9383 | 0.9198 | 0.9507 | 0.9360 | 0.9389 | 0.9368 ± 0.0111 |
| Inference (ms / 1k rows) | 3.3 | 2.4 | 1.5 | 1.7 | 1.8 | 2.1 ± 0.7 |
| Fit time (s) | 223.6 | 189.5 | 124.8 | 123.0 | 162.4 | 164.7 ± 43.1 |

### Training vs CV vs student-test

| Metric | Fold-train | CV mean | Full-train | Student-test |
|---|---|---|---|---|
| macro_f1 | 0.8944 | 0.8799 | 0.8925 | 0.8905 |
| top1_when_candidate | 0.8701 | 0.8461 | 0.8669 | 0.8573 |
| top1_all_points | 0.7737 | 0.7645 | 0.7699 | 0.8561 |
| route_continuity | 0.9279 | 0.9245 | 0.9273 | 0.9226 |
| log_loss | 0.1175 | 0.1354 | 0.1196 | 0.1151 |
| precision | 0.7015 | 0.6848 | 0.6975 | 0.6994 |
| recall | 0.9728 | 0.9368 | 0.9710 | 0.9501 |

Student-test confusion matrix `[[TN, FP], [FN, TP]]` = `[[173107, 7812], [955, 18178]]`.

### Interpretation

* **Best CV ranking of all models:** CV Top-1 0.8461 and route continuity 0.9245, but the lead
  over LightGBM (0.8431) and the MLP (0.8434) comes almost entirely from one fold (fold 3); see `docs/model_selection.md`.
* **Overfitting:** mild. Fold-train vs CV macro F1 0.894 vs 0.880, Top-1 0.870 vs
  0.846, log loss 0.117 vs 0.135: depth-15 trees fit the training trips more closely than the
  linear model, but the gap stays small because of min_samples_leaf = 10 and bagging.
* **Student-test** (macro F1 0.8905, Top-1 0.8573, log loss 0.1151) is in line with CV.
* **Cost:** the slowest model to train (202 s on all train trips, ~4 min per CV fold), which is why LightGBM, tied on
  Top-1, was chosen for Bayesian optimisation. Inference 2.38 ms per 1,000 rows; largest trip
  (00000084, 92,587 rows) 0.13 s.
