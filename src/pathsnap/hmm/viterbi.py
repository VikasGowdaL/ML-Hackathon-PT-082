"""Viterbi decoding for PathSnap candidate road arcs.

Inputs are candidate rows (obs_idx, arc_id, probability), the trip's arcs table,
and optionally a GPS-derived reset flag per observation. No ground-truth data
is accessed here.

This is a first-stage *arc-level* decoder. Predicted arc IDs are not yet a fully
connected traversed route: shortest-path bridging and sequence evaluation belong
in the subsequent integration stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ViterbiConfig:
    emission_weight: float = 1.0
    same_arc_bonus: float = 0.0
    connected_bonus: float = 0.0
    disconnected_penalty: float = 5.0
    probability_floor: float = 1e-9


def decode_viterbi(
    candidates: pd.DataFrame,
    arcs: pd.DataFrame,
    n_observations: Optional[int] = None,
    reset_at: Optional[np.ndarray] = None,
    config: Optional[ViterbiConfig] = None,
) -> pd.DataFrame:
    """Return one predicted road arc per observation where candidates exist.

    Args:
        candidates: Columns ``obs_idx``, ``arc_id``, ``probability`` (positive-class
            LightGBM probabilities). May be unsorted. Repeated (obs_idx, arc_id)
            pairs are not allowed.
        arcs: Columns ``arc_id``, ``from_node_id``, ``to_node_id``.
        n_observations: Total number of GPS observations (including those with no
            candidates). Defaults to max(obs_idx)+1.
        reset_at: Boolean array length n_observations; True forces a new sequence
            at that observation, e.g. when the GPS gap policy detects a break.
        config: Decoder parameters.

    Returns:
        DataFrame with ``obs_idx``, nullable ``predicted_arc_id``, ``status``.
        Missing observations have status ``no_candidates``. The Viterbi chain
        restarts after such observations or reset_at=True.
    """
    cfg = config or ViterbiConfig()
    if cfg.probability_floor <= 0 or cfg.probability_floor >= 1:
        raise ValueError("probability_floor must be in (0, 1)")
    if cfg.disconnected_penalty < 0:
        raise ValueError("disconnected_penalty must be nonnegative")

    needed_c = {"obs_idx", "arc_id", "probability"}
    needed_a = {"arc_id", "from_node_id", "to_node_id"}
    if not needed_c.issubset(candidates.columns):
        raise ValueError(f"candidates missing columns: {sorted(needed_c - set(candidates.columns))}")
    if not needed_a.issubset(arcs.columns):
        raise ValueError(f"arcs missing columns: {sorted(needed_a - set(arcs.columns))}")
    if arcs["arc_id"].duplicated().any():
        raise ValueError("arcs contains duplicate arc_id")
    if candidates[["obs_idx", "arc_id"]].duplicated().any():
        raise ValueError("candidates contains duplicate (obs_idx, arc_id)")

    if n_observations is None:
        n_observations = int(candidates["obs_idx"].max()) + 1 if len(candidates) else 0
    if n_observations < 0:
        raise ValueError("n_observations must not be negative")
    resets = np.zeros(n_observations, dtype=bool) if reset_at is None else np.asarray(reset_at, dtype=bool)
    if resets.shape != (n_observations,):
        raise ValueError("reset_at must have length n_observations")

    result = pd.DataFrame({
        "obs_idx": np.arange(n_observations, dtype=np.int64),
        "predicted_arc_id": pd.array([pd.NA] * n_observations, dtype="Int64"),
        "status": ["no_candidates"] * n_observations,
    })
    if candidates.empty:
        return result

    c = candidates.sort_values(["obs_idx", "arc_id"], kind="stable").copy()
    obs_idx = c["obs_idx"].to_numpy()
    arc_ids = c["arc_id"].to_numpy()
    p = c["probability"].to_numpy(dtype=float)
    if not np.issubdtype(obs_idx.dtype, np.integer) or (obs_idx < 0).any() or (obs_idx >= n_observations).any():
        raise ValueError("candidate obs_idx must be integers in [0, n_observations)")
    if not np.issubdtype(arc_ids.dtype, np.integer):
        raise ValueError("arc_id must be an integer")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must be finite and in [0, 1]")

    arc_map = arcs.set_index("arc_id")[["from_node_id", "to_node_id"]]
    if not pd.Index(arc_ids).isin(arc_map.index).all():
        raise ValueError("candidates contain arc IDs absent from arcs")
    if arc_map.isna().any().any():
        raise ValueError("arcs contain missing endpoint IDs")
    endpoint = arc_map.loc[arc_ids]
    starts = endpoint["from_node_id"].to_numpy(dtype=np.int64)
    ends = endpoint["to_node_id"].to_numpy(dtype=np.int64)

    # Grouped by observation. Each group records backpointers for backtracking.
    unique_obs, first, counts = np.unique(obs_idx, return_index=True, return_counts=True)
    groups = [slice(int(s), int(s + n)) for s, n in zip(first, counts)]
    chosen = {}
    chain = []
    last_scores = None
    previous_slice = None

    def flush():
        nonlocal chain, last_scores
        if not chain:
            return
        k = int(np.argmax(last_scores))
        for o, group, back in reversed(chain):
            chosen[o] = int(arc_ids[group][k])
            if back is not None:
                k = int(back[k])
        chain = []
        last_scores = None

    previous_obs = -2
    for obs, sl in zip(unique_obs, groups):
        obs = int(obs)
        emit = cfg.emission_weight * np.log(np.clip(p[sl], cfg.probability_floor, 1.0))
        restart = previous_slice is None or obs != previous_obs + 1 or resets[obs]
        if restart:
            flush()
            scores = emit.copy()
            back = None
        else:
            old = previous_slice
            same = arc_ids[old, None] == arc_ids[None, sl]
            connected = ends[old, None] == starts[None, sl]
            transition = np.where(
                same, cfg.same_arc_bonus,
                np.where(connected, cfg.connected_bonus, -cfg.disconnected_penalty),
            )
            score_matrix = last_scores[:, None] + transition
            back = np.argmax(score_matrix, axis=0)
            scores = emit + score_matrix[back, np.arange(score_matrix.shape[1])]
        chain.append((obs, sl, back))
        last_scores = scores
        previous_slice = sl
        previous_obs = obs
    flush()

    if chosen:
        idx = np.fromiter(chosen.keys(), dtype=np.int64)
        result.loc[idx, "predicted_arc_id"] = pd.array([chosen[int(i)] for i in idx], dtype="Int64")
        result.loc[idx, "status"] = "matched"
    return result
