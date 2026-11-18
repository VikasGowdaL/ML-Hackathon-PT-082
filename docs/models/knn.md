# Model Specification: K-Nearest Neighbors (KNN) Classifier

**Target Task**: Score candidate road arcs for GPS observations $(p(\text{label}=1 \mid \mathbf{x}))$.  
**Pipeline Location**: `src/pathsnap/models/train_knn.py`  
**Entry Point**: `model_factory(config=None)`, `run(trips=None)`, `subsample_training_fold()`

---

## 1. Rationale: Why KNN Suits Candidate Arc Scoring

- **Metric-Space Geometry**: Road arc candidate matching is fundamentally geometric. Candidates with near-zero perpendicular offset, aligned heading ($\Delta\theta \approx 0^\circ$), and high projection overlap occupy a compact cluster in standardized feature space.
- **Instance-Based Benchmark**: As a non-parametric model, KNN forms decision surfaces directly from similar stored observations without parametric functional assumptions.
- **Continuous Distance-Weighted Scoring**: By weighting neighbors by the inverse of their Euclidean distance in standardized feature space, KNN assigns higher likelihood to candidates exhibiting strong geometric proximity.

---

## 2. Expected Strengths and Limitations

### Strengths
- **Simplicity & Interpretability**: Predictions can be directly traced to specific nearest historical candidate pairs in the training fold.
- **Zero Training Phase**: Fitting is an index build ($\mathcal{O}(1)$ or $\mathcal{O}(N \log N)$), enabling rapid iteration.
- **Calibration with Inverse Distance**: Inverse distance weighting produces smoothly decaying probabilities as candidates deviate from known road profiles.

### Limitations & Computational Feasibility Strategy
- **Curse of Dimensionality**: In 20 dimensions, distances concentrate, and KD-tree / Ball-tree pruning degrades toward linear scan.
- **Inference Latency on Full Dataset**:
  - The 80 training trips contain **1,772,015 candidate rows**.
  - Querying 1.4 million training points for each test candidate takes ~5.5 seconds per 1,000 rows.
  - Across a validation fold of ~350,000 candidates, raw KNN inference would take over 30 minutes per fold.
- **Feasibility Solution (Negative Subsampling in Training Folds Only)**:
  - Inside each cross-validation fold, retain **100% of positive instances** (~148k rows).
  - Subsample negative instances to a controlled ratio (e.g., `negative_subsample_ratio=4.0`, or cap training rows to `max_train_samples=50,000`).
  - **Invariance Rule**: Subsampling occurs strictly within the training fold. Validation and test folds are **never subsampled** and evaluate across 100% of candidate observations.
  - At 50k training points, inference drops to ~38 ms per 1,000 rows, rendering 5-fold CV fully practical.

---

## 3. Initial Hyperparameters & Justifications

All hyperparameters are centralized in `KNN_CONFIG` in `src/pathsnap/models/train_knn.py`:

| Parameter | Initial Value | Justification |
|---|---|---|
| `n_neighbors` ($k$) | `21` | An odd $k=21$ avoids binary voting ties and produces continuous probabilities rather than coarse step fractions (e.g. $k=3 \to \{0, 0.33, 0.67, 1\}$), while retaining spatial locality. |
| `weights` | `"distance"` | Weights neighbor vote by $1 / d$. Closer points in standardized feature space dominate predictions, ensuring precision on tight geometric alignments. |
| `metric` | `"minkowski"` ($p=2$) | Standard Euclidean distance on standardized features ensures isotropic distance calculation across geometric dimensions. |
| `scaler` | `StandardScaler()` | Mandatory pipeline component. Normalizes differing feature dimensions (metres vs. angles vs. fractions) to zero mean and unit variance. |
| `imputer` | `SimpleImputer(strategy="median")` | Resolves the 0.10% NaNs present in heading angle features for stationary GPS points without data leakage. |
| `n_jobs` | `-1` | Parallelizes neighbor search across all CPU cores. |
| `subsample_negatives` | `True` (for CV) | Downsamples majority negatives in training folds to keep $\mathcal{O}(N_{\text{train}} \times N_{\text{val}})$ tractable. |
| `negative_subsample_ratio` | `4.0` | Keeps 4 negatives per 1 positive in training fold, preserving minority ground truth while reducing search space. |
| `random_state` | `42` | Ensures deterministic negative subsampling inside training folds. |

---

## 4. Objective & Probability Formulation

- **Distance Metric**:
  $$d(\mathbf{x}, \mathbf{z}) = \left(\sum_{j=1}^{20} \left(\frac{x_j - z_j}{\sigma_j}\right)^2 \right)^{1/2}$$
- **Probability Output**:
  $$\hat{p}(\text{label}=1 \mid \mathbf{x}) = \frac{\sum_{i \in \mathcal{N}_k(\mathbf{x}),\, y_i=1} w_i}{\sum_{i \in \mathcal{N}_k(\mathbf{x})} w_i}, \quad w_i = \frac{1}{d(\mathbf{x}, \mathbf{x}_i) + \epsilon}$$
- **Evaluation Metric (via Harness)**:
  - Binary Cross-Entropy (Log-Loss)
  - Precision-Recall AUC (PR-AUC)
  - Top-1 candidate ranking accuracy per GPS observation

---

## 5. Experimental Results

Run: `python -m pathsnap.models.train_knn` (default config: k = 21, distance weights, Euclidean, 4:1 negative subsampling capped at
50,000 training rows). Source: `outputs/results/experiments.csv` (model `knn`), `outputs/results/folds/knn_folds.csv`,
`outputs/results/student/knn_per_trip.csv`. PR-AUC and ROC-AUC are not computed by the shared harness and are therefore not reported.

### Grouped 5-fold CV (validation folds, 16 trips each)

| Metric | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean ± Std |
|---|---|---|---|---|---|---|
| Macro F1 | 0.8855 | 0.8736 | 0.8976 | 0.8750 | 0.8497 | 0.8763 ± 0.0177 |
| Log loss | 0.1603 | 0.1503 | 0.1358 | 0.1465 | 0.2193 | 0.1624 ± 0.0330 |
| Top-1 (true arc among candidates) | 0.8291 | 0.8034 | 0.8780 | 0.8129 | 0.7905 | 0.8228 ± 0.0339 |
| Top-1 (all points) | 0.8137 | 0.7785 | 0.5815 | 0.8114 | 0.7295 | 0.7429 ± 0.0964 |
| Route continuity | 0.9253 | 0.8961 | 0.9078 | 0.8474 | 0.9119 | 0.8977 ± 0.0300 |
| Precision | 0.7426 | 0.7213 | 0.7483 | 0.7225 | 0.6481 | 0.7166 ± 0.0401 |
| Recall | 0.8651 | 0.8335 | 0.9009 | 0.8360 | 0.8672 | 0.8605 ± 0.0275 |
| Inference (ms / 1k rows) | 60.8 | 72.9 | 58.1 | 60.4 | 53.7 | 61.2 ± 7.1 |
| Fit time (s) | 14.9 | 14.0 | 13.2 | 12.4 | 12.8 | 13.4 ± 1.0 |

### Training vs CV vs student-test

| Metric | Fold-train | CV mean | Full-train | Student-test |
|---|---|---|---|---|
| macro_f1 | 0.8919 | 0.8763 | 0.8905 | 0.8844 |
| top1_when_candidate | 0.8554 | 0.8228 | 0.8520 | 0.8451 |
| top1_all_points | 0.7606 | 0.7429 | 0.7567 | 0.8439 |
| route_continuity | 0.9081 | 0.8977 | 0.9079 | 0.9044 |
| log_loss | 0.1333 | 0.1624 | 0.1366 | 0.1338 |
| precision | 0.7375 | 0.7166 | 0.7362 | 0.7205 |
| recall | 0.8956 | 0.8605 | 0.8909 | 0.8829 |

Student-test confusion matrix `[[TN, FP], [FN, TP]]` = `[[174366, 6553], [2240, 16893]]`.

### Interpretation

* **Generalisation gap:** fold-train vs CV macro F1 0.892 vs 0.876 and Top-1
  0.855 vs 0.823: mild overfitting. Part of the training score is
  inflated because ~50,000 of the scored training rows are the stored neighbours themselves (distance weighting gives them
  their own label).
* **Ranking:** CV Top-1 0.823 is the lowest of the logged models (others 0.840–0.843), although row-level
  macro F1 (0.876) is high: the 4:1 subsampling shifts the threshold balance, but the neighbour vote ranks
  candidates of the same GPS point less consistently. Route continuity is also the lowest (0.898).
* **Cost:** inference 61.3 ms per 1,000 rows (≈ 30–60× the other models); the largest trip
  (00000084, 92,587 rows) takes 6.4 s. Training uses only
  50,000 of ~1.4 M rows (≈ 3.5 %).
* Weakest fold: fold 4 (contains trip 00000033), as for every other model.
