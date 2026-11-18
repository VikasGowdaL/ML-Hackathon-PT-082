
import { useEffect, useState } from "react";
import axios from "axios";
import {
  MapContainer,
  TileLayer,
  CircleMarker,
  Polyline,
  useMap,
} from "react-leaflet";
import {
  Satellite,
  Map,
  BarChart3,
  Route,
  Settings,
  Activity,
} from "lucide-react";
import "./App.css";

// Where the PathSnap API runs (two-laptop deployment: the API is on Laptop 1, this UI on Laptop 2).
// Priority: ?api=http://IP:PORT in the page URL (remembered in this browser)
//           > VITE_API_URL in frontend/.env.local
//           > a previously remembered ?api= value
//           > same host as this page, port 8000.
function resolveApiUrl() {
  const clean = (url) => url.trim().replace(/\/+$/, "");
  const fromQuery = new URLSearchParams(window.location.search).get("api");
  if (fromQuery) {
    try {
      localStorage.setItem("pathsnap_api_url", clean(fromQuery));
    } catch {
      /* storage unavailable: use it for this page load only */
    }
    return clean(fromQuery);
  }
  if (import.meta.env.VITE_API_URL) return clean(import.meta.env.VITE_API_URL);
  try {
    const saved = localStorage.getItem("pathsnap_api_url");
    if (saved) return saved;
  } catch {
    /* ignore */
  }
  return `http://${window.location.hostname}:8000`;
}

const API_URL = resolveApiUrl();

function FitMap({ points }) {
  const map = useMap();

  useEffect(() => {
    if (!points.length) return;

    const bounds = points.map((p) => [p.lat, p.lon]);

    if (bounds.length === 1) {
      map.setView(bounds[0], 16);
    } else {
      map.fitBounds(bounds, { padding: [35, 35] });
    }
  }, [points, map]);

  return null;
}

export default function App() {
  const [activeTab, setActiveTab] = useState("Dashboard");
  const [trips, setTrips] = useState([]);
  const [selectedTrip, setSelectedTrip] = useState("");
  const [tripData, setTripData] = useState(null);
  const [showGPS, setShowGPS] = useState(true);
  const [showTrack, setShowTrack] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [tripSource, setTripSource] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [predictError, setPredictError] = useState("");
  const [showRoute, setShowRoute] = useState(true);

  useEffect(() => {
    async function fetchTrips() {
      try {
        // Final evaluation: the reserved evaluation trips. Fallback: development trips.
        let available = [];
        try {
          const evalResponse = await axios.get(`${API_URL}/api/eval/trips`);
          available = evalResponse.data.trips || [];
          if (available.length > 0) {
            setTripSource(`Evaluation trips (${evalResponse.data.partition})`);
          }
        } catch {
          available = [];
        }
        if (available.length === 0) {
          const response = await axios.get(`${API_URL}/api/trips`);
          available = response.data.trips || [];
          setTripSource("Development trips (no evaluation trips on the server)");
        }

        setTrips(available);

        if (available.length > 0) {
          setSelectedTrip(available[0].trip_id);
        }
      } catch {
        setError(
          `Unable to connect to the PathSnap backend at ${API_URL}. ` +
            "Check that the API is running on Laptop 1 and that VITE_API_URL (or ?api=) points to it."
        );
      }
    }

    fetchTrips();
  }, []);

  useEffect(() => {
    if (!selectedTrip) return;

    let cancelled = false;

    async function fetchTrip() {
      setLoading(true);
      setError("");
      setTripData(null);
      setPrediction(null);
      setPredictError("");

      try {
        const response = await axios.get(
          `${API_URL}/api/trips/${selectedTrip}`
        );

        if (!cancelled) {
          setTripData(response.data);
        }
      } catch {
        if (!cancelled) {
          setError("Unable to load the selected trip.");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    fetchTrip();

    return () => {
      cancelled = true;
    };
  }, [selectedTrip]);

  const points = tripData?.gps_points || [];
  const positions = points.map((p) => [p.lat, p.lon]);
  const routeSections = prediction?.predicted_route || [];
  const metrics = prediction?.metrics;
  const pct = (v) => (v === undefined || v === null ? "—" : `${(100 * v).toFixed(1)}%`);

  async function runPrediction() {
    if (!selectedTrip) return;
    setPredicting(true);
    setPredictError("");
    try {
      const response = await axios.post(`${API_URL}/api/predict/${selectedTrip}`);
      setPrediction(response.data);
    } catch (err) {
      setPredictError(
        err?.response?.data?.detail || "Prediction failed. Is the API (and its model) available on Laptop 1?"
      );
    } finally {
      setPredicting(false);
    }
  }

  const menu = [
    { name: "Dashboard", icon: Map },
    { name: "Trips", icon: Route },
    { name: "Model Comparison", icon: BarChart3 },
    { name: "Settings", icon: Settings },
  ];

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <Satellite size={29} color="#38bdf8" />
          <span>PathSnap</span>
        </div>

        <div className="sidebar-caption">WORKSPACE</div>

        <nav>
          {menu.map(({ name, icon: Icon }) => (
            <button
              key={name}
              className={`nav-item ${
                activeTab === name ? "active" : ""
              }`}
              onClick={() => setActiveTab(name)}
            >
              <Icon size={19} />
              {name}
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <Activity size={17} color="#34d399" />
          Backend Integration
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <h1>{activeTab}</h1>
            <p>GPS Map Matching & Route Reconstruction</p>
          </div>

          <span className="demo-badge">
            {prediction ? "ROUTE PREDICTED" : tripData ? "REAL GPS DATA" : "AWAITING DATA"}
          </span>
        </header>

        <section className="stats-grid">
          <div className="stat-card">
            <span>Selected Trip</span>
            <strong>{selectedTrip || "None"}</strong>
          </div>

          <div className="stat-card">
            <span>GPS Observations</span>
            <strong>{tripData?.gps_count ?? 0}</strong>
          </div>

          <div className="stat-card">
            <span>Road Network Arcs</span>
            <strong>{tripData?.arcs_count ?? 0}</strong>
          </div>

          <div className="stat-card">
            <span>Dataset Partition</span>
            <strong>{tripData?.partition || "—"}</strong>
          </div>
        </section>

        <section className="map-panel">
          <div className="panel-header">
            <div>
              <h2>GPS Trip Visualization</h2>
              <p>
                {tripSource || "Select a trip"} — select a trip, then run the route prediction.
              </p>
            </div>

            <div className="map-controls">
              <select
                value={selectedTrip}
                onChange={(e) =>
                  setSelectedTrip(e.target.value)
                }
                style={{
                  background: "#0f172a",
                  color: "#fff",
                  padding: "10px",
                  border: "1px solid #334155",
                  borderRadius: "8px",
                }}
              >
                {trips.length === 0 && (
                  <option value="">No trips available</option>
                )}

                {trips.map((trip) => (
                  <option
                    key={trip.trip_id}
                    value={trip.trip_id}
                  >
                    {trip.trip_id} ({trip.partition})
                  </option>
                ))}
              </select>

              <button
                type="button"
                className="predict-button"
                onClick={runPrediction}
                disabled={!selectedTrip || predicting || loading}
              >
                {predicting ? "Predicting…" : "Predict Route"}
              </button>

              <label>
                <input
                  type="checkbox"
                  checked={showRoute}
                  onChange={(e) => setShowRoute(e.target.checked)}
                />
                Predicted Route
              </label>

              <label>
                <input
                  type="checkbox"
                  checked={showGPS}
                  onChange={(e) =>
                    setShowGPS(e.target.checked)
                  }
                />
                GPS Points
              </label>

              <label>
                <input
                  type="checkbox"
                  checked={showTrack}
                  onChange={(e) =>
                    setShowTrack(e.target.checked)
                  }
                />
                GPS Trace
              </label>
            </div>
          </div>

          {error && <p className="prediction-error">{error}</p>}

          {loading && <p>Loading GPS trip...</p>}
          {predicting && (
            <p className="prediction-notice">
              Laptop 1 is running candidate generation, ML scoring and Viterbi decoding for trip {selectedTrip}…
            </p>
          )}
          {predictError && <p className="prediction-error">{predictError}</p>}

          {!loading && !error && trips.length === 0 && (
            <p>No local GPS trips were found.</p>
          )}

          <div className="map-wrapper">
            <MapContainer
              center={[20, 0]}
              zoom={2}
              scrollWheelZoom={true}
              preferCanvas={true}
              style={{ height: "100%", width: "100%" }}
            >
              <TileLayer
                attribution="&copy; OpenStreetMap contributors"
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              />

              <FitMap points={points} />

              {showTrack && positions.length > 1 && (
                <Polyline
                  positions={positions}
                  pathOptions={{
                    color: "#22c55e",
                    weight: 3,
                  }}
                />
              )}

              {showGPS &&
                positions.map((point, index) => (
                  <CircleMarker
                    key={index}
                    center={point}
                    radius={4}
                    pathOptions={{
                      color: "#fff",
                      fillColor: "#38bdf8",
                      fillOpacity: 0.9,
                      weight: 1,
                    }}
                  />
                ))}

              {showRoute &&
                routeSections.map((section, i) => (
                  <Polyline
                    key={`route-${i}`}
                    positions={section}
                    pathOptions={{ color: "#a855f7", weight: 5, opacity: 0.9 }}
                  />
                ))}

              {positions.length > 0 && (
                <CircleMarker
                  center={positions[0]}
                  radius={9}
                  pathOptions={{ color: "#fff", fillColor: "#22c55e", fillOpacity: 1, weight: 2 }}
                />
              )}
              {positions.length > 1 && (
                <CircleMarker
                  center={positions[positions.length - 1]}
                  radius={9}
                  pathOptions={{ color: "#fff", fillColor: "#ef4444", fillOpacity: 1, weight: 2 }}
                />
              )}
            </MapContainer>
          </div>

          <div className="map-legend">
            <span>
              <i className="legend-dot gps" />
              GPS Observations
            </span>
            <span>
              <i className="legend-dot route" />
              GPS Trace (not map-matched)
            </span>
            <span>
              <i className="legend-dot predicted" />
              Predicted Route (ML + HMM)
            </span>
            <span>
              <i className="legend-dot start" />
              Start
            </span>
            <span>
              <i className="legend-dot end" />
              End
            </span>
          </div>

          {prediction && (
            <div className="prediction-panel">
              <div className="prediction-heading">
                <h2>Prediction Results — trip {prediction.trip_id}</h2>
                <span>
                  {prediction.partition} · {prediction.gps_count} GPS points
                  {prediction.cached ? " · cached" : ""}
                </span>
              </div>
              {metrics ? (
                <section className="prediction-metrics">
                  <div className="prediction-metric">
                    <span>Route accuracy (F1, by length)</span>
                    <strong>{pct(metrics.route_f1)}</strong>
                  </div>
                  <div className="prediction-metric">
                    <span>Precision / Recall</span>
                    <strong>
                      {pct(metrics.route_precision)} / {pct(metrics.route_recall)}
                    </strong>
                  </div>
                  <div className="prediction-metric">
                    <span>GPS points on true route (ML + HMM)</span>
                    <strong>{pct(metrics.points_on_route)}</strong>
                  </div>
                  <div className="prediction-metric">
                    <span>GPS points on true route (ML only)</span>
                    <strong>{pct(metrics.points_on_route_ml_only)}</strong>
                  </div>
                  <div className="prediction-metric">
                    <span>Predicted / true route length</span>
                    <strong>
                      {metrics.predicted_route_km.toFixed(2)} / {metrics.reference_route_km.toFixed(2)} km
                    </strong>
                  </div>
                  <div className="prediction-metric">
                    <span>Route sections / gaps</span>
                    <strong>
                      {metrics.route_sections} / {metrics.unbridged_breaks}
                    </strong>
                  </div>
                  <div className="prediction-metric">
                    <span>Processing time</span>
                    <strong>
                      {prediction.timing_s.total_s.toFixed(2)} s{prediction.cached ? " (cached)" : ""}
                    </strong>
                  </div>
                  <div className="prediction-metric">
                    <span>Predicted arcs</span>
                    <strong>{prediction.predicted_arc_count}</strong>
                  </div>
                </section>
              ) : (
                <p>No reference route is available on the server for this trip, so no accuracy is shown.</p>
              )}
              <p className="prediction-notice" style={{ color: "#94a3b8", marginTop: 14 }}>
                Model: {prediction.model.estimator} · Decoder: {prediction.model.decoder}. The reference route is
                used only by the server to compute these scores after prediction.
              </p>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
