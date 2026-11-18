import { useEffect, useRef, useState } from "react";
import axios from "axios";
import {
  MapContainer,
  TileLayer,
  CircleMarker,
  Polyline,
  GeoJSON,
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

// Laptop 1 supplies the API. Laptop 2 only renders the UI.
function resolveApiUrl() {
  const clean = (url) => url.trim().replace(/\/+$/, "");
  const fromQuery = new URLSearchParams(window.location.search).get("api");
  if (fromQuery) {
    try {
      localStorage.setItem("pathsnap_api_url", clean(fromQuery));
    } catch {
      /* Storage is optional. */
    }
    return clean(fromQuery);
  }
  if (import.meta.env.VITE_API_URL) return clean(import.meta.env.VITE_API_URL);
  try {
    const saved = localStorage.getItem("pathsnap_api_url");
    if (saved) return saved;
  } catch {
    /* Ignore unavailable storage. */
  }
  return `http://${window.location.hostname}:8000`;
}

const API_URL = resolveApiUrl();

function FitMap({ points }) {
  const map = useMap();
  useEffect(() => {
    if (!points.length) return;
    const bounds = points.map((p) => [p.lat, p.lon]);
    if (bounds.length === 1) map.setView(bounds[0], 16);
    else map.fitBounds(bounds, { padding: [35, 35] });
  }, [points, map]);
  return null;
}

// Expected prediction response shape is documented in API_CONTRACT.md.
// Only accept actual geographic GeoJSON -- never interpret arc IDs as coordinates.
function getGeoJSON(payload) {
  const geojson = payload?.route_geojson ?? payload?.geojson ?? payload?.route;
  if (!geojson || typeof geojson !== "object") return null;
  if (geojson.type === "FeatureCollection" && Array.isArray(geojson.features)) {
    return geojson;
  }
  if (geojson.type === "Feature" && geojson.geometry) {
    return { type: "FeatureCollection", features: [geojson] };
  }
  return null;
}

function isLineFeature(feature) {
  return feature?.geometry?.type === "LineString" || feature?.geometry?.type === "MultiLineString";
}

function getErrorMessage(error) {
  if (error?.response?.status === 404 || error?.response?.status === 405) {
    return "Prediction endpoint is not available on Laptop 1 yet. Ask the backend teammate to implement GET /api/predict/{trip_id}.";
  }
  if (error?.response?.data?.detail) {
    const detail = error.response.data.detail;
    return typeof detail === "string" ? detail : JSON.stringify(detail);
  }
  if (error?.code === "ECONNABORTED") return "Prediction timed out. Check the Laptop 1 server logs.";
  return "Could not request a prediction from Laptop 1. Check the server, network, and API configuration.";
}

function Metric({ label, value }) {
  if (value === undefined || value === null) return null;
  return (
    <div className="prediction-metric">
      <span>{label}</span>
      <strong>{String(value)}</strong>
    </div>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState("Dashboard");
  const [trips, setTrips] = useState([]);
  const [selectedTrip, setSelectedTrip] = useState("");
  const [tripData, setTripData] = useState(null);
  const [showGPS, setShowGPS] = useState(true);
  const [showTrack, setShowTrack] = useState(true);
  const [showPrediction, setShowPrediction] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [predictionError, setPredictionError] = useState("");
  const [predictionLoading, setPredictionLoading] = useState(false);
  const [predictionRequestedTrip, setPredictionRequestedTrip] = useState("");

  useEffect(() => {
    let cancelled = false;
    async function fetchTrips() {
      try {
        const response = await axios.get(`${API_URL}/api/trips`);
        const available = Array.isArray(response.data.trips) ? response.data.trips : [];
        if (!cancelled) {
          setTrips(available);
          setSelectedTrip(available[0]?.trip_id ?? "");
        }
      } catch {
        if (!cancelled) {
          setError(`Unable to connect to the PathSnap backend at ${API_URL}. Check the Laptop 1 server and API URL.`);
        }
      }
    }
    fetchTrips();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    // Changing trips invalidates every previously loaded prediction and request.
    setPrediction(null);
    setPredictionError("");
    setPredictionLoading(false);
    setPredictionRequestedTrip("");
    if (!selectedTrip) {
      setTripData(null);
      return;
    }

    let cancelled = false;
    async function fetchTrip() {
      setLoading(true);
      setError("");
      setTripData(null);
      try {
        const response = await axios.get(`${API_URL}/api/trips/${encodeURIComponent(selectedTrip)}`);
        if (!cancelled) setTripData(response.data);
      } catch {
        if (!cancelled) setError("Unable to load the selected trip.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    fetchTrip();
    return () => { cancelled = true; };
  }, [selectedTrip]);

  // Use request generation ID so an old response can never overwrite a newer trip.
  const [predictionRequestId, setPredictionRequestId] = useState(0);
  const latestTrip = useRef(selectedTrip);
  latestTrip.current = selectedTrip;
  async function runPrediction() {
    if (!selectedTrip || predictionLoading) return;
    const requestTrip = selectedTrip;
    const requestId = predictionRequestId + 1;
    setPredictionRequestId(requestId);
    setPredictionLoading(true);
    setPrediction(null);
    setPredictionError("");
    setPredictionRequestedTrip(requestTrip);
    try {
      const response = await axios.get(`${API_URL}/api/predict/${encodeURIComponent(requestTrip)}`, {
        timeout: 180000,
      });
      if (latestTrip.current === requestTrip) {
        setPrediction(response.data);
      }
    } catch (err) {
      if (latestTrip.current === requestTrip) setPredictionError(getErrorMessage(err));
    } finally {
      if (latestTrip.current === requestTrip) setPredictionLoading(false);
    }
  }

  const points = tripData?.gps_points || [];
  const positions = points.map((p) => [p.lat, p.lon]);
  const resultForSelectedTrip = predictionRequestedTrip === selectedTrip ? prediction : null;
  const allGeoJSON = getGeoJSON(resultForSelectedTrip);
  const routeGeoJSON = allGeoJSON && {
    type: "FeatureCollection",
    features: allGeoJSON.features.filter(isLineFeature),
  };
  const metrics = resultForSelectedTrip?.metrics || {};

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
              className={`nav-item ${activeTab === name ? "active" : ""}`}
              onClick={() => setActiveTab(name)}
            >
              <Icon size={19} />
              {name}
            </button>
          ))}
        </nav>
        <div className="sidebar-footer">
          <Activity size={17} color="#34d399" />
          Laptop 2 · API: {API_URL}
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <h1>{activeTab}</h1>
            <p>GPS Map Matching &amp; Route Reconstruction</p>
          </div>
          <span className="demo-badge">{tripData ? "REAL GPS DATA" : "AWAITING DATA"}</span>
        </header>

        <section className="stats-grid">
          <div className="stat-card"><span>Selected Trip</span><strong>{selectedTrip || "None"}</strong></div>
          <div className="stat-card"><span>GPS Observations</span><strong>{tripData?.gps_count ?? 0}</strong></div>
          <div className="stat-card"><span>Road Network Arcs</span><strong>{tripData?.arcs_count ?? 0}</strong></div>
          <div className="stat-card"><span>Dataset Partition</span><strong>{tripData?.partition || "—"}</strong></div>
        </section>

        <section className="map-panel">
          <div className="panel-header">
            <div>
              <h2>GPS Trip &amp; Predicted Route</h2>
              <p>Select a trip, then request its map-matched route from Laptop 1.</p>
            </div>
            <div className="map-controls">
              <select
                aria-label="Select trip"
                value={selectedTrip}
                onChange={(e) => setSelectedTrip(e.target.value)}
                style={{ background: "#0f172a", color: "#fff", padding: "10px", border: "1px solid #334155", borderRadius: "8px" }}
              >
                {trips.length === 0 && <option value="">No trips available</option>}
                {trips.map((trip) => (
                  <option key={trip.trip_id} value={trip.trip_id}>
                    {trip.trip_id} ({trip.partition})
                  </option>
                ))}
              </select>
              <label><input type="checkbox" checked={showGPS} onChange={(e) => setShowGPS(e.target.checked)} /> GPS Points</label>
              <label><input type="checkbox" checked={showTrack} onChange={(e) => setShowTrack(e.target.checked)} /> GPS Trace</label>
              <label><input type="checkbox" checked={showPrediction} onChange={(e) => setShowPrediction(e.target.checked)} /> Predicted Route</label>
              <button type="button" className="predict-button" disabled={!selectedTrip || predictionLoading} onClick={runPrediction}>
                {predictionLoading ? "Predicting…" : "Run Prediction"}
              </button>
            </div>
          </div>

          {error && <p className="prediction-error">{error}</p>}
          {loading && <p>Loading GPS trip...</p>}
          {!loading && !error && trips.length === 0 && <p>No GPS trips returned by Laptop 1.</p>}
          {predictionLoading && <p className="prediction-notice">Laptop 1 is running inference for trip {predictionRequestedTrip}…</p>}
          {predictionError && <p className="prediction-error">{predictionError}</p>}
          {resultForSelectedTrip && !routeGeoJSON && (
            <p className="prediction-error">Prediction returned, but no GeoJSON FeatureCollection was found. Check the API response contract.</p>
          )}

          <div className="map-wrapper">
            <MapContainer center={[20, 0]} zoom={2} scrollWheelZoom={true} style={{ height: "100%", width: "100%" }}>
              <TileLayer attribution="&copy; OpenStreetMap contributors" url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
              <FitMap points={points} />
              {showTrack && positions.length > 1 && (
                <Polyline positions={positions} pathOptions={{ color: "#22c55e", weight: 3 }} />
              )}
              {showPrediction && routeGeoJSON && routeGeoJSON.features.length > 0 && (
                <GeoJSON
                  key={`${selectedTrip}-${predictionRequestId}`}
                  data={routeGeoJSON}
                  style={(feature) => ({
                    color: feature?.properties?.source === "bridge" ? "#fb923c" : "#a855f7",
                    weight: feature?.properties?.source === "bridge" ? 5 : 4,
                    opacity: 0.95,
                  })}
                />
              )}
              {showGPS && positions.map((point, index) => (
                <CircleMarker key={index} center={point} radius={4} pathOptions={{ color: "#fff", fillColor: "#38bdf8", fillOpacity: 0.9, weight: 1 }} />
              ))}
            </MapContainer>
          </div>
          <div className="map-legend">
            <span><i className="legend-dot gps" /> GPS Observations</span>
            <span><i className="legend-dot route" /> GPS Trace (not map-matched)</span>
            <span><i className="legend-dot predicted" /> Predicted Route</span>
            <span><i className="legend-dot bridge" /> Bridging Arcs</span>
          </div>
        </section>

        {resultForSelectedTrip && (
          <section className="prediction-panel">
            <div className="prediction-heading">
              <h2>Prediction Results · Trip {selectedTrip}</h2>
              <span>{routeGeoJSON?.features.length ?? 0} rendered route sections</span>
            </div>
            <div className="prediction-metrics">
              <Metric label="Observation Accuracy" value={metrics.observation_accuracy} />
              <Metric label="Route Continuity" value={metrics.route_continuity} />
              <Metric label="Matched Observations" value={metrics.matched_observations} />
              <Metric label="Bridge Arcs" value={metrics.bridges_inserted} />
              <Metric label="Unbridgeable Pairs" value={metrics.unbridgeable_pairs} />
              <Metric label="Route Sections" value={metrics.route_sections} />
              <Metric label="Inference Time (ms)" value={metrics.inference_time_ms} />
            </div>
            {Object.keys(metrics).length === 0 && (
              <p className="prediction-notice">No evaluation metrics were returned by Laptop 1.</p>
            )}
          </section>
        )}
      </main>
    </div>
  );
}
