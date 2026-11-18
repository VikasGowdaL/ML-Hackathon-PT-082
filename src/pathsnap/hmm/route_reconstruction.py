"""Reconstruct a directed route from observation-level matched arc IDs.

This is post-processing: it does not modify Viterbi emission/transition scores.
Unreachable pairs are recorded as breaks, never silently joined.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import networkx as nx


@dataclass
class ReconstructionResult:
    route_arcs: list[int]
    breaks: list[tuple[int, int]]
    bridges_inserted: int
    total_bridge_length_m: float


def reconstruct_route(matched_arcs: Iterable[int], graph: nx.MultiDiGraph,
                      max_bridge_m: float | None = 2000.0) -> ReconstructionResult:
    """Connect consecutive matched directed arcs using minimum-length road paths.

    The route retains repeated arcs when they are consecutive observations on the
    same arc only once. For distinct arcs, the bridge starts at the previous arc's
    end node and ends at the next arc's start node. Parallel edges are resolved
    by minimum ``length_m`` then ``arc_id``. ``breaks`` contains (previous, next)
    arc IDs that could not be connected under the optional distance cutoff.

    Note: output is a list of arcs with possible breaks; do not render as a
    single continuous polyline when ``breaks`` is nonempty.
    """
    if max_bridge_m is not None and max_bridge_m < 0:
        raise ValueError("max_bridge_m must be nonnegative or None")
    ids = [int(a) for a in matched_arcs]
    if not ids:
        return ReconstructionResult([], [], 0, 0.0)
    endpoints = graph.graph.get("arc_endpoints", {})
    for arc in ids:
        if arc not in endpoints:
            raise ValueError(f"Arc {arc} is absent from the graph")

    result = [ids[0]]
    breaks: list[tuple[int, int]] = []
    inserted = 0
    total_m = 0.0
    cache: dict[tuple[int, int], tuple[list[int], float] | None] = {}

    for prev, nxt in zip(ids, ids[1:]):
        if prev == nxt:
            continue
        end = endpoints[prev][1]
        start = endpoints[nxt][0]
        bridge: tuple[list[int], float] | None
        if end == start:
            bridge = ([], 0.0)
        elif (end, start) in cache:
            bridge = cache[(end, start)]
        else:
            try:
                nodes = nx.shortest_path(graph, end, start, weight="length_m")
                arcs: list[int] = []
                length = 0.0
                for u, v in zip(nodes, nodes[1:]):
                    key, attrs = min(
                        graph[u][v].items(),
                        key=lambda item: (float(item[1].get("length_m", float("inf"))), int(item[0])),
                    )
                    arcs.append(int(key))
                    length += float(attrs["length_m"])
                bridge = (arcs, length)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                bridge = None
            cache[(end, start)] = bridge

        if bridge is None or (max_bridge_m is not None and bridge[1] > max_bridge_m):
            breaks.append((prev, nxt))
        else:
            result.extend(bridge[0])
            inserted += len(bridge[0])
            total_m += bridge[1]
        result.append(nxt)

    return ReconstructionResult(result, breaks, inserted, total_m)
