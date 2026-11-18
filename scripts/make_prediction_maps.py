#!/usr/bin/env python3
"""Folium maps of the full pipeline: GPS, road network, reference route and predicted route.

Usage:
    python scripts/make_prediction_maps.py                       # 3 representative student_test trips
    python scripts/make_prediction_maps.py --trips 00000000 00000077

For each trip, runs the final pipeline (pathsnap.pipeline.predict_trip; the reference route is not used for
prediction), loads the reference route afterwards, and writes outputs/maps/prediction_<trip_id>.html with
toggleable layers:

* Road network     - grey, arcs within ~300 m of the GPS track's bounding box
* GPS observations - blue dots (all points up to 3,000, otherwise a seeded sample) + thin trace
* Reference route  - green (ground truth, student_test only)
* Predicted route  - purple (final model + Viterbi + shortest-path reconstruction)
* Layer order (bottom to top): road network, GPS, reference route, predicted route; start / end markers,
  legend, metrics box, layer control

Only train / student_test trips are allowed (evaluator routes are never drawn into committed maps).
"""

from __future__ import annotations

import argparse

import folium
import numpy as np

from pathsnap import pipeline as P
from pathsnap.road_network import load_split
from pathsnap.utils.config import get_repo_root

DEFAULT_TRIPS = ["00000000", "00000091", "00000077"]  # best, typical, hardest student_test trip (pipeline results)
MAX_POINTS = 3000
SEED = 42


def _arc_lines(arc_ids, nodes, arcs):
    lon, lat = nodes["lon"].to_numpy(), nodes["lat"].to_numpy()
    f, t = arcs["from_node_id"].to_numpy(), arcs["to_node_id"].to_numpy()
    return [[[float(lat[f[a]]), float(lon[f[a]])], [float(lat[t[a]]), float(lon[t[a]])]] for a in arc_ids]


def make_map(trip_id: str, partition: str) -> tuple:
    pred = P.predict_trip(trip_id)
    ref = P.load_reference_route(trip_id, partition)
    metrics = P.score_prediction(pred, ref)
    obs, nodes, arcs = pred.observations, pred.nodes, pred.arcs

    lat0, lat1, lon0, lon1 = obs.lat.min(), obs.lat.max(), obs.lon.min(), obs.lon.max()
    pad_lat, pad_lon = 0.003, 0.003 / max(np.cos(np.radians(obs.lat.mean())), 0.2)  # ~300 m
    nlon, nlat = nodes["lon"].to_numpy(), nodes["lat"].to_numpy()
    inside = (nlat >= lat0 - pad_lat) & (nlat <= lat1 + pad_lat) & (nlon >= lon0 - pad_lon) & (nlon <= lon1 + pad_lon)
    f, t = arcs["from_node_id"].to_numpy(), arcs["to_node_id"].to_numpy()
    network_arcs = np.flatnonzero(inside[f] & inside[t])

    m = folium.Map(location=[obs.lat.mean(), obs.lon.mean()], zoom_start=14, tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[lat0, lon0], [lat1, lon1]])

    net = folium.FeatureGroup(name=f"Road network ({len(network_arcs):,} arcs near the trip)", show=True)
    folium.PolyLine(_arc_lines(network_arcs, nodes, arcs), color="#9ca3af", weight=1, opacity=0.6).add_to(net)
    net.add_to(m)

    idx = np.arange(len(obs))
    if len(idx) > MAX_POINTS:
        idx = np.sort(np.random.default_rng(SEED).choice(idx, size=MAX_POINTS, replace=False))
    gps = folium.FeatureGroup(name=f"GPS observations ({len(idx):,} of {len(obs):,} shown)")
    folium.PolyLine(obs[["lat", "lon"]].to_numpy().tolist(), color="#38bdf8", weight=1, opacity=0.5).add_to(gps)
    for i in idx:
        r = obs.iloc[i]
        folium.CircleMarker([r.lat, r.lon], radius=1.5, color="#0284c7", fill=True, fill_opacity=0.6, weight=0,
                            tooltip=f"point {i}, t={r.t:.0f}s").add_to(gps)
    gps.add_to(m)

    refg = folium.FeatureGroup(name=f"Reference route ({len(set(ref.tolist()))} arcs)")
    folium.PolyLine(_arc_lines(sorted(set(ref.tolist())), nodes, arcs), color="#16a34a", weight=7, opacity=0.55,
                    tooltip="Reference (ground-truth) route").add_to(refg)
    refg.add_to(m)

    predg = folium.FeatureGroup(name=f"Predicted route ({len(pred.sections)} section(s))")
    for i, line in enumerate(P.route_polylines(pred)):
        folium.PolyLine(line, color="#a855f7", weight=4, opacity=0.95, tooltip=f"Predicted route, section {i + 1}").add_to(predg)
    predg.add_to(m)

    folium.Marker([obs.lat.iloc[0], obs.lon.iloc[0]], tooltip="GPS start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([obs.lat.iloc[-1], obs.lon.iloc[-1]], tooltip="GPS end", icon=folium.Icon(color="red", icon="stop")).add_to(m)

    legend = f"""
    <div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:10px 12px;border:1px solid #888;
                border-radius:6px;font:12px/1.5 Arial,sans-serif">
      <b>PathSnap – trip {trip_id} ({partition})</b><br>
      <span style="color:#9ca3af">&#9644;</span> road network &nbsp;
      <span style="color:#0284c7">&#9679;</span> GPS observations<br>
      <span style="color:#16a34a">&#9644;</span> reference route &nbsp;
      <span style="color:#a855f7">&#9644;</span> predicted route (ML + HMM)<br>
      route F1 {100 * metrics['route_f1']:.1f}% · precision {100 * metrics['route_precision']:.1f}% ·
      recall {100 * metrics['route_recall']:.1f}%<br>
      GPS points on true route: {100 * metrics['points_on_route']:.1f}% (ML only {100 * metrics['points_on_route_ml_only']:.1f}%)
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    return m, metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trips", nargs="+", default=DEFAULT_TRIPS)
    args = ap.parse_args()
    split = load_split()
    part_of = {t: p for p in ("train", "student_test") for t in split[p]}
    out = get_repo_root() / "outputs" / "maps"
    out.mkdir(parents=True, exist_ok=True)
    for tid in args.trips:
        if tid not in part_of:
            raise SystemExit(f"{tid}: only train / student_test trips can be mapped with their reference route")
        m, s = make_map(tid, part_of[tid])
        path = out / f"prediction_{tid}.html"
        m.save(str(path))
        print(f"{tid}: route F1 {s['route_f1']:.4f}, points on route {s['points_on_route']:.4f} -> {path.relative_to(get_repo_root()).as_posix()}")


if __name__ == "__main__":
    main()
