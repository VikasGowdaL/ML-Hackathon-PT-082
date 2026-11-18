# Data Quality & Decision Log

This log records observed anomalies, edge cases, coordinate transformations, and data sanitization decisions across all 100 trips in PathSnap.

---

## Partition Structure & Wording
- **Total Dataset**: 100 trips in 3 partitions: **80 train, 10 student-test, and the evaluator 10 held by the organisers**.
- **Current Raw Ingestion**: 80 training trips (`data/raw/train/`) and 10 student-testing trips (`data/raw/student_test/`).
- **Evaluator Guard**: The evaluator trips are held by the organisers for the final demo and are not in our possession. The evaluator guard in `get_trips()` is built for that future evaluation case to strictly protect against accidental leakage.
- **Naming Convention**: Uniformly standardized to snake_case across all configs, inventory, and code:
  - `train`
  - `student_test`
  - `evaluator_test`

---

## Observed Anomalies & Findings

### [2026-11-18] - Massive Time Gaps (Vehicle Pauses / Signal Losses)
- **Trips Affected**: 
  - `trip_00000090`: Max time gap $\Delta t_{\max} = 153,064\text{ s}$ (~42.5 hours, 536 GPS points total).
  - `trip_00000078`: Max time gap $\Delta t_{\max} = 52,810\text{ s}$ (~14.7 hours, 2,909 GPS points total).
  - `trip_00000069`: Max time gap $\Delta t_{\max} = 2,658\text{ s}$ (~44.3 minutes, 2,067 GPS points total).
  - `trip_00000021`: Max time gap $\Delta t_{\max} = 2,161\text{ s}$ (~36.0 minutes, 15,783 GPS points total).
- **Category**: Temporal Sampling Discontinuity
- **Impact on Modeling**:
  - Uninterrupted sequence models assume continuous vehicle motion between consecutive points. Over multi-hour pauses, shortest-path road distance and $\Delta t$ become meaningless.
- **Decision & Guidance**:
  - In feature extraction and candidate generation, split trajectories into contiguous sub-trajectories or reset sequence states whenever $\Delta t > 30\text{ s}$.

### [2026-11-18] - Fractional Timestamps in Trip 71
- **Trip Affected**: `trip_00000071`
- **Category**: Timestamp Data Type & Granularity
- **Observation**:
  - Trip 71 exhibits non-integer timestamps: `[1.0, 2.0, 7.2, 10.2, 11.2, 12.2...]` with fractional 0.2s offsets.
- **Root Cause / Impact**:
  - Asynchronous GPS receiver clock drift or sub-second logging. If timestamps are cast to integer, time delta resolution is degraded and artificial 0-second gaps appear.
- **Decision & Guidance**:
  - Explicitly load and retain `t` as `np.float64` in `_read_track()` and `TripData.track`. Never cast `t` to integer.

### [2026-11-18] - Very Large Road Networks
- **Trips Affected**:
  - `trip_00000013`: 307,392 arcs, 155,003 nodes (largest road network in dataset; ~1.95 MB `.arcs`, ~1.69 MB `.nodes`).
  - `trip_00000032`: 259,437 arcs, 124,540 nodes.
  - `trip_00000086`: 237,656 arcs, 114,868 nodes.
  - `trip_00000043`: 215,620 arcs, 109,070 nodes.
  - `trip_00000055`: 198,961 arcs, 98,511 nodes (~75 km north-south span).
- **Category**: Graph Memory & Spatial Search Complexity
- **Impact on Modeling**:
  - Naive global spatial searches (computing point-to-arc distance for all 300,000 arcs) will cause severe memory and latency bottlenecks.
- **Decision & Guidance**:
  - Spatial indexing (e.g. `scipy.spatial.cKDTree` or `shapely.STRtree`) is mandatory for candidate generation. Restrict candidate arc queries to a spatial bounding radius around each GPS point (e.g. $r = 50\text{ m}$).

### [2026-11-18] - Coordinate Projection & UTM Zone Boundary Crossings
- **Trips Affected**:
  - `trip_00000011`: Crosses 60°E boundary between UTM Zone 40 and Zone 41 ($lon \in [59.637^\circ, 60.026^\circ]$).
  - `trip_00000097`: Crosses 12°E boundary between UTM Zone 32 and Zone 33 ($lon \in [11.855^\circ, 12.231^\circ]$).
  - `trip_00000013` and `trip_00000055`: Cover wide regional networks (~75 km bounding spans).
- **Category**: Spatial Projection & Coordinate Reference Systems
- **Observation**:
  - Euclidean distance on raw $(lon, lat)$ degrees produces non-metric, latitude-dependent distortions because $1^\circ$ longitude equals $\approx 111.32 \cdot \cos(\text{lat}) \text{ km}$.
- **Decision & Mathematical Rationale**:
  - A per-trip UTM zone is chosen dynamically using the bounding box center (`compute_utm_epsg()`).
  - For boundary-crossing trips (trips 11 and 97), projecting all coordinates to a single central UTM zone maintains a continuous, metric Cartesian plane across the entire trip and road network. Transverse Mercator projections remain mathematically smooth beyond the 6° zone boundary (scale distortion remains $< 0.1\%$ within 0.5° of the edge, less than 1 meter error over 1 km).

### [2026-11-18] - Disk Cache Policy & Invalidation
- **Category**: Performance & Reproducibility
- **Decision**:
  - Cached files in `data/cache/` are git-ignored.
  - Every cached `TripData` embeds `metadata["loader_version"] = LOADER_VERSION` (currently `1.2.0`). If the loader code or schema version changes, cached objects are automatically invalidated and re-read from raw data.
  - Added programmatic `clear_cache()` and CLI flag `scripts/build_inventory.py --clear-cache`.
