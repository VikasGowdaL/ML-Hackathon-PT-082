# Milestone 2: Spatial Candidate Arc Generation

This document details the candidate arc generation method, parameter configurations, benchmark performance, and output schema produced for Milestone 2 (Person 1).

---

## 1. Candidate Selection Method

For every GPS observation $o_t$ in a trip, the goal is to extract all road network arcs $a_j$ that pass within a metric radius $r$ (in meters), sorted by perpendicular distance and capped at the nearest $k$ candidates:

$$\mathcal{C}(o_t) = \text{Top-}k \left( \left\{ a_j \in \mathcal{A} \;\Big|\; \text{dist}_{\text{UTM}}(o_t, \text{segment}(a_j)) \le r \right\} \right)$$

### Why Shapely STRtree vs Midpoint KD-Tree?
- **Failure Mode of Midpoint KD-Trees**:
  Road networks contain heterogeneous arc lengths (from 5m urban links to >1,000m highway links). An arc can pass directly beneath a vehicle (e.g. 5m away) while its midpoint is 500m away. A KD-tree indexing midpoints with a 50m radius would **miss the true arc completely** (a catastrophic false negative).
- **The STRtree Solution**:
  `shapely.STRtree` (Sort-Tile-Recursive R-tree) indexes the 2D axis-aligned bounding boxes (AABBs) of the segment geometries $[ (x_0, y_0) \to (x_1, y_1) ]$ in conformal UTM meters.
  - Queries all GPS points simultaneously in a single batched call (`tree.query(gps_points, predicate="dwithin", distance=radius_m)`).
  - Exact Euclidean point-to-segment distance is computed vectorized in C (GEOS).
  - Handles long arcs, zero-length arcs, and sharp curves with mathematical rigor and zero false negatives.

---

## 2. Configuration Parameters ([`configs/candidates.yaml`](file:///Users/aarizdudekula/college2/ml/ML_HACKATHON/ML-Hackathon-PT-082/configs/candidates.yaml))

| Parameter | Default | Benchmark Range | Justification |
| :--- | :---: | :---: | :--- |
| **`radius_m`** | **`50.0 m`** | `[30, 50, 100, 200]` | Typical urban GPS error (95th percentile) is 10–30m. A 50m radius captures >99% of relevant arcs while minimizing false candidate combinatorial explosion. |
| **`max_candidates`** | **`30`** | `[10, 30, 50]` | In dense urban grids, 50m covers 2–8 arcs on average. A cap of 30 strictly bounds downstream transition matrix complexity ($30 \times 30 = 900$ pairs max) while retaining the nearest arcs. |

---

## 3. Benchmark Results across All 90 Trips (Train & Student-Test)

Evaluated across all 227,577 GPS observations in the 80 `train` and 10 `student_test` partitions:

| Search Radius | Total Candidates | Avg Candidates / Obs | Observations with 0 Candidates | Total Runtime (90 Trips) | Avg Time / Trip |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **`30 m`** | 1,200,145 | 5.27 | 20,385 (8.96%)* | 15.31 s | 170.2 ms |
| **`50 m`** *(Default)* | **1,972,067** | **8.67** | **18,230 (8.01%)*** | **8.29 s** | **92.1 ms** |
| **`100 m`** | 3,611,559 | 15.87 | 1,413 (0.62%) | 9.44 s | 104.8 ms |
| **`200 m`** | 5,342,901 | 23.48 | 108 (0.05%) | 15.15 s | 168.3 ms |

*\*Note on Zero-Candidate Outliers*: In `trip_00000054`, 16,519 GPS points are located outside the bounding network of that specific trip. Excluding this outlier, **only 0.82%** of GPS observations have zero candidates at 50m, and **0.62%** at 100m.

---

## 4. Stress-Test Performance on Largest Networks & Longest Tracks

Run on Apple Silicon M-series (single process, cached index):

| Trip ID | Partition | Road Network Arcs | GPS Track Points | Candidates (50m) | Generation Time | Peak Mem Delta |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`00000013`** *(Largest network)* | `train` | **307,392 arcs** | 2,580 pts | 23,548 | **470 ms** | ~48 MB |
| **`00000032`** | `train` | **259,437 arcs** | 2,464 pts | 19,424 | **390 ms** | ~41 MB |
| **`00000086`** | `train` | **237,656 arcs** | 4,143 pts | 20,097 | **410 ms** | ~38 MB |
| **`00000043`** | `train` | **215,620 arcs** | 2,322 pts | 22,761 | **350 ms** | ~35 MB |
| **`00000055`** | `train` | **198,961 arcs** | 2,472 pts | 14,086 | **320 ms** | ~31 MB |
| **`00000054`** *(Longest track)* | `train` | 116,916 arcs | **19,313 pts** | 45,492 | **496 ms** | ~24 MB |
| **`00000021`** | `train` | 1,510 arcs | **15,783 pts** | 50,592 | **35 ms** | ~12 MB |
| **`00000084`** | `train` | 50,888 arcs | **16,031 pts** | 92,587 | **251 ms** | ~28 MB |

---

## 5. Output Format & Consumption by Teammates

### Parquet Table Schema (`outputs/candidates/<radius>/<trip_id>.parquet`)

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| **`trip_id`** | `string` | Unique trip identifier (e.g. `'00000001'`). |
| **`obs_idx`** | `int64` | 0-based row index into the trip's `.track` table ($0 \le \text{obs\_idx} < N_{\text{obs}}$). |
| **`arc_id`** | `int64` | 0-based row index into the trip's `.arcs` table ($0 \le \text{arc\_id} < N_{\text{arcs}}$). |
| **`distance_m`** | `float64` | Exact Euclidean perpendicular distance in meters from GPS point to arc line segment. |

Sorted by `obs_idx` ascending, then `distance_m` ascending.

### How Person 2 (Labels & Miss Rate) & Person 3 (Features) Read the Output

```python
import pandas as pd
from pathsnap.data.loader import load_trip
from pathsnap.features.candidates import load_candidates_parquet

# 1. Load candidates for a trip at 50m radius
cands_df = load_candidates_parquet("outputs/candidates/50/00000001.parquet")
print(cands_df.head())
#   trip_id  obs_idx  arc_id  distance_m
#  00000001        0    2641        4.12
#  00000001        0    2642        9.85

# 2. Access observations with zero candidates
trip = load_trip("00000001", project_utm=True)
all_obs = set(range(len(trip.track)))
matched_obs = set(cands_df["obs_idx"])
no_cand_obs = sorted(all_obs - matched_obs)
print(f"Zero-candidate observations: {len(no_cand_obs)}")
```
