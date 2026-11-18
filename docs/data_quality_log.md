# Data Quality & Decision Log

This log records observed anomalies, edge cases, data sanitization steps, and design decisions across all 100 trips in PathSnap.

---

## Log Template

### [YYYY-MM-DD] - Entry Title
- **Trip ID(s) Affected**: (e.g. `trip_042` or `All`)
- **Category**: (Missing Files | Timestamp Jumps | Invalid Coordinates | Dangling Node IDs | Disconnected Graph)
- **Observation**:
  - Detailed description of what was observed during data loading or inventory generation.
- **Root Cause / Impact**:
  - Why this occurred and how it impacts HMM emission/transition modeling.
- **Decision & Action Taken**:
  - Specific fix applied in code (e.g. imputation, filtering, or handling in `loader.py`).
- **Owner**: (Person 1 / 2 / 3 / 4)

---

## Initial Entries

### [2026-11-18] - Coordinate Projection Decision: UTM over Degree Euclidean
- **Trip ID(s) Affected**: All trips
- **Category**: Coordinate Representation & Distance Metric
- **Observation**:
  - Calculating Euclidean distances directly on $(lon, lat)$ degrees produces non-metric, latitude-dependent distortions because $1^\circ$ longitude equals $\approx 111.32 \cdot \cos(\text{lat}) \text{ km}$.
- **Root Cause / Impact**:
  - HMM emission probabilities model GPS noise using metric standard deviations (e.g. $\sigma_{\text{GPS}} \approx 10\text{ m}$). Road network edge lengths also require accurate meter measurements for transition probabilities.
- **Decision & Action Taken**:
  - Implemented dynamic UTM projection via `pyproj` based on each trip's longitude/latitude bounding box (`compute_utm_epsg()` and `project_coordinates()` in `src/pathsnap/data/loader.py`). Coordinates are converted to conformal Cartesian meters $(x, y)$.
- **Owner**: Person 1

### [2026-11-18] - Evaluator Leakage Guard Implementation
- **Trip ID(s) Affected**: Evaluator partition trips (10 trips)
- **Category**: Data Leakage Prevention
- **Observation**:
  - The evaluator partition contains ground-truth `.route` files that must not leak into model training or validation.
- **Root Cause / Impact**:
  - Accidental inclusion of evaluator labels in training or tuning invalidates final hackathon benchmark results.
- **Decision & Action Taken**:
  - In `src/pathsnap/data/loader.py`, `load_trip` and `get_trips` strictly block loading `.route` for the evaluator partition unless `allow_evaluator_labels=True` is explicitly passed. A high-priority warning is logged whenever evaluator labels are accessed.
- **Owner**: Person 1


### [2026-11-18] - Dataset Inventory Analysis & Sampling Gap Findings
- **Trip ID(s) Affected**: All 90 delivered trips (80 `train`, 10 `student-test`)
- **Category**: Dataset Structure, Integrity & Temporal Discontinuities
- **Observation**:
  - Raw dataset verified: 80 trips in `data/raw/train/` and 10 trips in `data/raw/student_test/`. The 10 evaluator trips are reserved for final evaluation.
  - Zero missing files: All 90 trips have valid `.track`, `.nodes`, `.arcs`, and `.route` files.
  - Median GPS sampling interval is strictly 1.0 second (~1 Hz).
  - 42 out of 90 trips exhibit sampling gaps $> 10\text{ s}$. Extreme pauses observed in `trip_00000090` ($\Delta t_{\max} = 153,064\text{ s} \approx 42.5\text{ hrs}$) and `trip_00000078` ($\Delta t_{\max} = 52,810\text{ s} \approx 14.6\text{ hrs}$).
- **Root Cause / Impact**:
  - Long GPS interruptions correspond to vehicle parking, engine-off pauses, or tunnel signal outages.
  - Feeding a single uninterrupted HMM chain across hours of missing GPS points breaks standard transition probabilities $p(d_t \mid \Delta t)$, which assume continuous vehicle motion along the shortest road network path.
- **Decision & Action Taken**:
  - Saved full dataset inventory to `outputs/inventory/trip_inventory.csv`.
  - Registered 80 train trips and 10 student-test trips in `configs/split.json`.
  - Recommendation for HMM module (Person 2/3): split trajectories into independent sub-trajectories or re-initialize emission priors when $\Delta t > 30\text{ s}$.
- **Owner**: Person 1
