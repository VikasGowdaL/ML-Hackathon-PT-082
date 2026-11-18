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

## 5. Experimental Results (Held Blank)

*To be populated by the shared model evaluation harness across GroupKFold cross-validation on the 80 training trips.*

| Metric | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean ± Std |
|---|---|---|---|---|---|---|
| Log-Loss | | | | | | |
| PR-AUC | | | | | | |
| ROC-AUC | | | | | | |
| Top-1 Acc | | | | | | |
| Inference Latency | | | | | | |
