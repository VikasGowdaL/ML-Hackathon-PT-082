
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

  useEffect(() => {
    async function fetchTrips() {
      try {
        const response = await axios.get(`${API_URL}/api/trips`);
        const available = response.data.trips || [];

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
            {tripData ? "REAL GPS DATA" : "AWAITING DATA"}
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
                Select a trip to visualize its actual GPS
                observations.
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

          {error && (
            <p style={{ color: "#f87171" }}>{error}</p>
          )}

          {loading && <p>Loading GPS trip...</p>}

          {!loading && !error && trips.length === 0 && (
            <p>No local GPS trips were found.</p>
          )}

          <div className="map-wrapper">
            <MapContainer
              center={[20, 0]}
              zoom={2}
              scrollWheelZoom={true}
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
          </div>
        </section>
      </main>
    </div>
  );
}
