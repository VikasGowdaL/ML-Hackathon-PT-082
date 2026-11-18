# PathSnap 🛰️🗺️
> **Map-matching noisy GPS trips to a road network and HMM/Viterbi route reconstruction**

PathSnap reconstructs high-fidelity road routes from sparse, noisy GPS trajectory points (~1 Hz) by projecting geographic coordinates to metric UTM space, matching candidates to road network arcs, and decoding sequence trajectories using Hidden Markov Models (HMM) and Viterbi dynamic programming.

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
  *(If PowerShell displays an execution policy error, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*
- **Windows (Command Prompt)**:
  ```cmd
  .venv\Scripts\activate.bat
  ```

### Manual Setup (Alternative)
If you prefer setting up manually:

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

---

## 2. Configuring Raw Data Location (`RAW_DATA_DIR`)

You can place raw trip data in the default directory `data/raw/` (git-ignored) or point to an external folder without modifying code:

- **Option A: Environment Variable (Recommended for different machines)**
  - **macOS / Linux (zsh/bash)**:
    ```bash
    export RAW_DATA_DIR="/path/to/my/raw/data"
    ```
  - **Windows (PowerShell)**:
    ```powershell
    $env:RAW_DATA_DIR = "C:\path\to\my\raw\data"
    ```
  - **Windows (CMD)**:
    ```cmd
    set RAW_DATA_DIR=C:\path\to\my\raw\data
    ```

- **Option B: `configs/paths.yaml`**
  Edit `configs/paths.yaml`:
  ```yaml
  raw_data_dir: "data/raw" # or "/absolute/path/to/raw/data"
  ```

---

## 3. Dataset Inventory and Splitting

To scan raw trips, compute temporal/spatial statistics, generate the 80/10/10 split, and produce `outputs/inventory/trip_inventory.csv`:

```bash
# Run with default or configured raw data directory
python scripts/build_inventory.py

# Or specify a custom raw data path directly
python scripts/build_inventory.py --raw-dir /path/to/raw/data --seed 42
```

Outputs created:
- `outputs/inventory/trip_inventory.csv`: One row per trip containing point counts, time gaps, bounding box, network size, and missing file checks.
- `configs/split.json`: Strictly disjoint partitions (`train`: 80 trips, `student-test`: 10 trips, `evaluator-testing`: 10 trips). Contains **trip IDs only** (never labels).

---

## 4. Leakage Rules & Data Governance

1. **Strict Partition Disjointness**: A trip ID must never appear in more than one partition.
2. **Evaluator Isolation**: Evaluator data and `.route` ground truth labels must **NEVER** be used for training, feature extraction, model selection, or tuning.
3. **Student Code Isolation**: Evaluator ground truth must stay completely out of any student-side prediction code, CLI tools, or web UI.
4. **Preprocessing Guard**: All learned parameters (e.g. noise distributions, speed priors, normalizations) must be fit on training trips only.
5. **Programmatic Guard**:
   ```python
   from pathsnap.data.loader import get_trips

   # Safe: route labels are withheld by default for evaluator-testing
   eval_trips = get_trips("evaluator-testing", allow_evaluator_labels=False)
   assert eval_trips[0].route is None
   ```

---

## 5. Repository Directory Layout

```
ML-Hackathon-PT-082/
├── configs/
│   ├── paths.yaml              # Central path configuration
│   └── split.json              # 80/10/10 train/student-test/evaluator trip IDs
├── data/
│   ├── README.md               # Raw file format specifications and partition rules
│   ├── raw/                    # (Git-ignored) Local raw trip data (.track, .nodes, .arcs, .route)
│   └── cache/                  # (Git-ignored) Fast serialized parquet/pickle caches
├── docs/
│   └── data_quality_log.md     # Anomaly and decision log
├── notebooks/                  # EDA, map visualizations, error analysis notebooks
├── outputs/
│   ├── inventory/              # trip_inventory.csv
│   ├── maps/                   # Folium HTML trajectory overlays
│   ├── plots/                  # Visual evaluation plots
│   └── results/                # Evaluation metrics and experiments.csv
├── scripts/
│   └── build_inventory.py      # Scan dataset, compute stats, and manage split
├── src/
│   └── pathsnap/
│       ├── api/                # FastAPI service endpoints
│       ├── data/               # loader.py, UTM projection, validation, caching
│       ├── features/           # Spatial, temporal, and candidate arc feature engineering
│       ├── hmm/                # Emission/transition probabilities and Viterbi decoder
│       ├── models/             # Candidate ranking models (LightGBM/XGBoost/heuristics)
│       └── utils/              # config.py, experiment_log.py
├── tests/                      # pytest test suite
├── .gitignore                  # Comprehensive gitignore for ML & cross-platform artifacts
├── pyproject.toml              # Modern Python packaging (pip install -e .)
├── requirements.txt            # Pinned cross-platform dependencies
├── setup_env.py                # Pure Python cross-platform venv setup
└── README.md                   # This document
```

---

## 6. Running Tests

```bash
pytest -v
```
All unit tests verify loader contracts, coordinate projections to UTM, partition disjointness, and evaluator label guards.
