# PathSnap Raw Data Specification & Split

## 1. Raw Data Storage Policy
- **Never commit raw data to Git!**
- Place raw data files under `data/raw/` (which is included in `.gitignore`) or specify an external directory using the `RAW_DATA_DIR` environment variable or `configs/paths.yaml`.
- Parsed and cached files reside in `data/cache/` (also git-ignored).

## 2. File Format per Trip
Each trip comprises up to 4 files sharing a common trip identifier:

| File | Header | Format | Content | Key Invariants |
| :--- | :--- | :--- | :--- | :--- |
| **`.track`** | None | 3 whitespace-separated columns (`lon`, `lat`, `t`) | GPS track points sampled at ~1 Hz | `lon, lat` in WGS84 degrees, `t` timestamp as float in seconds |
| **`.nodes`** | None | 2 whitespace-separated columns (`lon`, `lat`) | Road network vertex coordinates | **0-based row index = `node_id`** |
| **`.arcs`** | None | 2 whitespace-separated int columns (`from_node_id`, `to_node_id`) | Directed road graph edges | **0-based row index = `arc_id`**. Node IDs must exist in `.nodes`. |
| **`.route`** | None | 1 integer column (`arc_id`) | Chronological ground-truth arc sequence | Arc IDs must exist in `.arcs`. **Evaluator route is secret.** |

> **IMPORTANT**:
> - A `.track` row does **NOT** correspond 1-to-1 with a `.nodes` or `.arcs` row. GPS points are noisy observations of motion along the road network.
> - Not all trips are identical in duration, density, or bounding box. Always verify row counts, dtypes, and coordinate validity.

## 3. Dataset Partitions (100 Trips Total)
The dataset comprises **80 train, 10 student-test, and the evaluator 10 held by the organisers**:
1. **Training Partition (`train`, 80 trips)**: Located in `data/raw/train/`. Used for candidate generation, label extraction, exploratory analysis, and model training.
2. **Student-Testing Partition (`student_test`, 10 trips)**: Located in `data/raw/student_test/`. Used for offline benchmark evaluations, miss rate computation, and feature validation.
3. **Evaluator-Testing Partition (`evaluator_test`, 10 trips)**: Held by the organisers for the final evaluation demo. We do not possess these trips; the guard in `get_trips()` is implemented for that future case.

### Leakage Rules
- **Disjointness**: A trip ID must never appear in more than one partition.
- **Evaluator Isolation**: Evaluator data and labels (`.route`) must NEVER be used for training, feature extraction, model selection, or tuning.
- **Student Code Isolation**: Evaluator ground truth must stay strictly out of student-side prediction code and UI.
- **Preprocessing Fitting**: All learned scalers, noise models, or priors must be fit solely on the training partition (`train`).
- **Evaluator Guard**: `pathsnap.data.loader.get_trips("evaluator_test")` blocks loading `.route` ground truth by default and logs an explicit warning if unmasked.
