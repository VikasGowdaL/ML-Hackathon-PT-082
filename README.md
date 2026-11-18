# PathSnap 🛰️🗺️
> **Map-matching noisy GPS trips to a road network and route reconstruction**

PathSnap reconstructs high-fidelity road routes from sparse, noisy GPS trajectory points (~1 Hz) by projecting geographic coordinates to metric UTM space, generating candidate road arcs, extracting geometric and sequence features, and decoding ground-truth trajectories.

---

## 1. Cross-Platform Environment Setup

Both **macOS** and **Windows** are fully supported. No C/C++ compiler is required on Windows; all dependencies use pre-built binary wheels.

### Automated Setup (Recommended)
From the repository root, run the cross-platform setup script with Python 3.10, 3.11, or 3.12:

```bash
# macOS / Linux
python3 setup_env.py

# Windows (PowerShell or CMD)
python setup_env.py
```
This automatically creates `.venv`, upgrades `pip`, installs `requirements.txt`, and installs `pathsnap` in editable mode (`pip install -e .`).

### Activation Commands

- **macOS / Linux (zsh or bash)**:
  ```bash
  source .venv/bin/activate
  ```
- **Windows (PowerShell)**:
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
- **Windows (Command Prompt)**:
  ```cmd
  .venv\Scripts\activate.bat
  ```

---

## 2. Dataset Partitions & Leakage Governance

The benchmark comprises 100 trips in 3 partitions:
- **80 train** (`train`): Located in `data/raw/train/` (80 trip subdirectories `00000001` through `00000099`).
- **10 student-test** (`student_test`): Located in `data/raw/student_test/` (10 trip subdirectories `00000000`, `00000007`, `00000009`, `00000040`, `00000052`, `00000056`, `00000065`, `00000076`, `00000077`, `00000091`).
- **The evaluator 10 held by the organisers** (`evaluator_test`): We do not have the evaluator trips; they will be provided for the final evaluation demo only. The guard in `get_trips("evaluator_test")` is implemented for that future case to strictly withhold `.route` ground truth.

Partition names are uniformly standardized with underscores: `train`, `student_test`, and `evaluator_test`.

### Leakage Rules
1. **Strict Partition Disjointness**: A trip ID must never appear in more than one partition.
2. **Evaluator Isolation**: Evaluator data and `.route` ground truth labels must **NEVER** be used for training, feature extraction, model selection, or tuning.
3. **Student Code Isolation**: Evaluator ground truth must stay completely out of any student-side prediction code or UI.
4. **Preprocessing Guard**: All learned parameters (e.g. noise distributions, scalers, candidate thresholds) must be fit on training trips only.
5. **Programmatic Guard**:
   ```python
   from pathsnap.data.loader import get_trips

   # Safe: route labels are withheld by default for evaluator_test
   eval_trips = get_trips("evaluator_test", allow_evaluator_labels=False)
   ```

---

## 3. Dataset Inventory and Splitting

To scan raw trips, compute temporal/spatial statistics, and verify `configs/split.json`:

```bash
# Run inventory builder
python scripts/build_inventory.py

# Optional: purge cache before building
python scripts/build_inventory.py --clear-cache
```

Outputs:
- `outputs/inventory/trip_inventory.csv`: One row per trip with file sizes, point counts, time gaps, bounding box, network sizes, and fractional timestamp flags.
- `configs/split.json`: Strictly disjoint partitions (`train`, `student_test`, `evaluator_test`). Contains **trip IDs only** (zero labels).

---

## 4. Repository Directory Layout

```text
ML-Hackathon-PT-082/
├── configs/
│   ├── paths.yaml              # Central path configuration
│   └── split.json              # 80 train / 10 student_test / evaluator_test trip IDs
├── data/
│   ├── README.md               # Raw file format specifications and partition rules
│   ├── raw/                    # (Git-ignored) Local raw trip data (train/ and student_test/)
│   └── cache/                  # (Git-ignored) Version-invalidated serialized caches
├── docs/
│   └── data_quality_log.md     # Anomaly, projection, and data quality log
├── notebooks/                  # EDA, map visualizations, error analysis notebooks
├── outputs/
│   ├── inventory/              # trip_inventory.csv
│   ├── maps/                   # Trajectory overlays
│   ├── plots/                  # Visual evaluation plots
│   └── results/                # Evaluation metrics and experiments.csv
├── scripts/
│   └── build_inventory.py      # Scan dataset, compute stats, and manage split
├── src/
│   └── pathsnap/
│       ├── api/                # Future milestone: Serving & API endpoints
│       ├── data/               # loader.py, UTM projection, validation, caching
│       ├── features/           # Milestone 2: Geometric & sequence features
│       ├── hmm/                # Future milestone: HMM transition & emission decoders
│       ├── models/             # Milestone 2: Candidate ranking & scoring
│       └── utils/              # config.py, experiment_log.py
├── tests/                      # pytest test suite
├── .gitignore                  # Gitignore for raw data, caches, and build artifacts
├── pyproject.toml              # Modern Python packaging (pip install -e .)
├── requirements.txt            # Pinned cross-platform dependencies
├── setup_env.py                # Pure Python cross-platform venv setup
└── README.md                   # This document
```

---

## 5. Milestone 2 Split & Next Steps

Following the project plan, Milestone 2 focuses on candidate extraction and feature engineering:

1. **Candidate Generation**: Spatial indexing (`cKDTree` / `STRtree`) to query candidate road arcs within radius $r$ around GPS points in metric UTM coordinates.
2. **Labels and Miss Rate**: Ground-truth label assignment from `.route` to candidate arcs, and calculation of candidate recall / miss rate on `train` and `student_test`.
3. **Geometry Features**: Perpendicular distances, heading alignment, arc lengths, and projected point offsets.
4. **Sequence Features plus Assembly**: Consecutive GPS delta times ($\Delta t$), speed estimates, topological transition connectivity between candidate arcs, and assembling tabular datasets for candidate ranking.

*(HMM transition/emission decoding and API deployment are scheduled for subsequent milestones).*
