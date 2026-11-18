"""Evaluate baseline LightGBM + arc-level Viterbi on one student-test trip.

Run: PYTHONPATH=src python -m pathsnap.hmm.predict_route --trip-id 00000000
No evaluator labels or routes are loaded by this module.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pathsnap.models.harness import FEATURES
from pathsnap.road_network import load_road_network, build_road_graph
from pathsnap.hmm.route_reconstruction import reconstruct_route
from pathsnap.hmm.viterbi import decode_viterbi, ViterbiConfig


def evaluate(trip_id: str, model_path: Path, output_dir: Path) -> None:
    # Read unlabelled candidate features; labels are joined separately for evaluation.
    source = Path('outputs/features/feature_table_student_test.parquet')
    columns = ['trip_id', 'obs_idx', 'arc_id', 'after_gap', 'is_segment_start'] + FEATURES
    columns = list(dict.fromkeys(columns))
    table = pd.read_parquet(source, columns=columns)
    table = table.loc[table.trip_id.astype(str) == trip_id].copy()
    if table.empty:
        raise ValueError(f'No candidate features found for student-test trip {trip_id}')

    model = joblib.load(model_path)
    features = table[FEATURES].astype(np.float32).to_numpy()
    table['probability'] = model.predict_proba(features)[:, 1]
    arcs_nodes, arcs, report = load_road_network(trip_id, scan=False)
    if arcs is None:
        raise RuntimeError(f'Could not load arcs: {report["errors"]}')

    # Derive observation count from student-test alignment; it is NOT passed to the model.
    labels = pd.read_parquet('outputs/features/student_test_labels.parquet')
    labels = labels.loc[labels.trip_id.astype(str) == trip_id, ['obs_idx', 'reference_arc_id']].copy()
    if labels.empty:
        raise ValueError(f'Missing student-test evaluation labels for {trip_id}')
    n_obs = int(labels.obs_idx.max()) + 1
    reset = np.zeros(n_obs, dtype=bool)
    gap_flags = table.groupby('obs_idx')[['after_gap', 'is_segment_start']].max()
    for obs, row in gap_flags.iterrows():
        if 0 <= int(obs) < n_obs:
            reset[int(obs)] = bool(row['after_gap']) or bool(row['is_segment_start'])

    decoded = decode_viterbi(
        table[['obs_idx', 'arc_id', 'probability']], arcs,
        n_observations=n_obs, reset_at=reset, config=ViterbiConfig(),
    )
    # Independent candidate argmax, with one row per observation.
    independent = table.loc[table.groupby('obs_idx')['probability'].idxmax(), ['obs_idx', 'arc_id']]
    independent = independent.rename(columns={'arc_id': 'top1_arc_id'})
    comparison = labels.merge(independent, on='obs_idx', how='left', validate='one_to_one')
    comparison = comparison.merge(decoded, on='obs_idx', how='left', validate='one_to_one')
    for name, column in [('LightGBM Top-1', 'top1_arc_id'), ('LightGBM + Viterbi', 'predicted_arc_id')]:
        hits = comparison[column].eq(comparison['reference_arc_id']).fillna(False)
        print(f'{name:20s}: {hits.sum()}/{len(comparison)} = {hits.mean():.4%} observation accuracy')

    # Simple directed consecutive-arc continuity, not full-route continuity.
    endpoint = arcs.set_index('arc_id')[['from_node_id', 'to_node_id']]
    for name, column in [('LightGBM Top-1', 'top1_arc_id'), ('LightGBM + Viterbi', 'predicted_arc_id')]:
        ids = comparison[column].tolist()
        connected, total = 0, 0
        for a, b in zip(ids, ids[1:]):
            if pd.isna(a) or pd.isna(b):
                continue
            a, b = int(a), int(b)
            if a not in endpoint.index or b not in endpoint.index:
                continue
            total += 1
            connected += int(a == b or endpoint.at[a, 'to_node_id'] == endpoint.at[b, 'from_node_id'])
        print(f'{name:20s}: {connected}/{total} = {connected / total:.4%} adjacent-arc continuity' if total else f'{name}: no comparable transitions')

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f'{trip_id}_viterbi_comparison.csv'
    comparison.to_csv(path, index=False)
    print(f'Saved comparison: {path}')

    # Reconstruct directed routes without crossing observation gaps or resets.
    graph, _, _ = build_road_graph(arcs_nodes, arcs)

    route_rows = []
    break_rows = []
    subroute = -1
    previous_arc = None
    previous_obs = None
    inserted = 0
    bridge_length = 0.0

    for row in comparison.sort_values("obs_idx").itertuples(index=False):
        obs = int(row.obs_idx)
        arc_value = row.predicted_arc_id

        if pd.isna(arc_value):
            previous_arc = None
            previous_obs = None
            continue

        arc = int(arc_value)
        start_new = (
            previous_arc is None
            or previous_obs is None
            or obs != previous_obs + 1
            or reset[obs]
        )

        if start_new:
            subroute += 1
            route_rows.append({
                "subroute": subroute,
                "arc_id": arc,
                "source": "matched",
            })
        else:
            result = reconstruct_route(
                [previous_arc, arc],
                graph,
                max_bridge_m=2000.0,
            )

            if result.breaks:
                break_rows.append({
                    "previous_obs": previous_obs,
                    "next_obs": obs,
                    "previous_arc": previous_arc,
                    "next_arc": arc,
                })
                subroute += 1
                route_rows.append({
                    "subroute": subroute,
                    "arc_id": arc,
                    "source": "matched",
                })
            else:
                # The first arc is already in the output.
                bridge_arcs = result.route_arcs[1:-1]

                for bridge_arc in bridge_arcs:
                    route_rows.append({
                        "subroute": subroute,
                        "arc_id": bridge_arc,
                        "source": "bridge",
                    })

                if arc != previous_arc:
                    route_rows.append({
                        "subroute": subroute,
                        "arc_id": arc,
                        "source": "matched",
                    })

                inserted += result.bridges_inserted
                bridge_length += result.total_bridge_length_m

        previous_arc = arc
        previous_obs = obs

    route_path = output_dir / f"{trip_id}_reconstructed_arcs.csv"
    breaks_path = output_dir / f"{trip_id}_route_breaks.csv"

    pd.DataFrame(
        route_rows,
        columns=["subroute", "arc_id", "source"]
    ).to_csv(route_path, index=False)

    pd.DataFrame(
        break_rows,
        columns=["previous_obs", "next_obs", "previous_arc", "next_arc"]
    ).to_csv(breaks_path, index=False)

    print("\\nShortest-path reconstruction:")
    print(f"Bridge arcs inserted : {inserted}")
    print(f"Total bridge distance: {bridge_length:.2f} m")
    print(f"Unbridgeable pairs   : {len(break_rows)}")
    print(f"Route sections       : {subroute + 1}")
    print(f"Saved route          : {route_path}")
    print(f"Saved breaks         : {breaks_path}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trip-id', default='00000000')
    parser.add_argument('--model', type=Path, default=Path('artifacts/models/lightgbm.joblib'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/results/hmm'))
    args = parser.parse_args()
    evaluate(args.trip_id, args.model, args.output_dir)


if __name__ == '__main__':
    main()
