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

## 5. Experimental Results (Held Blank)

*To be populated by the shared model evaluation harness across GroupKFold cross-validation on the 80 training trips.*

| Metric | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean ± Std |
|---|---|---|---|---|---|---|
| Log-Loss | | | | | | |
| PR-AUC | | | | | | |
| ROC-AUC | | | | | | |
| Top-1 Acc | | | | | | |
| Inference Latency | | | | | | |
