
import { useState } from "react";
import {
  MapContainer,
  TileLayer,
  Polyline,
  CircleMarker,
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

const sampleRoute = [
  [12.9716, 77.5946],
  [12.9731, 77.5971],
  [12.9752, 77.6004],
  [12.9778, 77.6032],
  [12.9802, 77.6061],
];

const gpsPoints = [
  [12.9718, 77.5942],
  [12.9729, 77.5975],
  [12.9756, 77.6001],
  [12.9775, 77.6036],
  [12.9805, 77.6058],
];

function FitRoute({ points }) {
  const map = useMap();

  useState(() => {
    // MapContainer starts at the sample route's center.
  });

  return null;
}

export default function App() {
  const [activeTab, setActiveTab] = useState("Dashboard");
  const [showGPS, setShowGPS] = useState(true);
  const [showRoute, setShowRoute] = useState(true);

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
          Frontend Demo Mode
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <h1>{activeTab}</h1>
            <p>GPS Map Matching & Route Reconstruction</p>
          </div>
          <span className="demo-badge">DEMO DATA</span>
        </header>

        <section className="stats-grid">
          <div className="stat-card">
            <span>Selected Trip</span>
            <strong>Demo Trip 001</strong>
          </div>
          <div className="stat-card">
            <span>GPS Observations</span>
            <strong>{gpsPoints.length}</strong>
          </div>
          <div className="stat-card">
            <span>Route Points</span>
            <strong>{sampleRoute.length}</strong>
          </div>
          <div className="stat-card">
            <span>Mode</span>
            <strong>Visualization</strong>
          </div>
        </section>

        <section className="map-panel">
          <div className="panel-header">
            <div>
              <h2>Route Visualization</h2>
              <p>GPS observations and demonstration route</p>
            </div>

            <div className="map-controls">
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
                  checked={showRoute}
                  onChange={(e) =>
                    setShowRoute(e.target.checked)
                  }
                />
                Route
              </label>
            </div>
          </div>

          <div className="map-wrapper">
            <MapContainer
              center={[12.9752, 77.6004]}
              zoom={14}
              scrollWheelZoom={true}
              style={{ height: "100%", width: "100%" }}
            >
              <TileLayer
                attribution='&copy; OpenStreetMap contributors'
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              />

              {showRoute && (
                <Polyline
                  positions={sampleRoute}
                  pathOptions={{
                    color: "#22c55e",
                    weight: 5,
                  }}
                />
              )}

              {showGPS &&
                gpsPoints.map((point, index) => (
                  <CircleMarker
                    key={index}
                    center={point}
                    radius={6}
                    pathOptions={{
                      color: "#fff",
                      fillColor: "#38bdf8",
                      fillOpacity: 1,
                      weight: 2,
                    }}
                  />
                ))}
            </MapContainer>
          </div>

          <div className="map-legend">
            <span>
              <i className="legend-dot gps" /> GPS Observations
            </span>
            <span>
              <i className="legend-dot route" /> Demo Route
            </span>
          </div>
        </section>
      </main>
    </div>
  );
}
