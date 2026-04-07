const map = L.map("map").setView([22.5, 78.9], 5);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 10,
  attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);

let plannedLine = null;
let actualLine = null;
let currentMarker = null;
let anomalyMarkers = [];

const engineBadge = document.getElementById("engineBadge");
const airspaceBadge = document.getElementById("airspaceBadge");

const pathMetrics = document.getElementById("pathMetrics");
const engineMetrics = document.getElementById("engineMetrics");
const airspaceMetrics = document.getElementById("airspaceMetrics");
const newsList = document.getElementById("newsList");
const clock = document.getElementById("clock");

const controlIds = ["route_deviation", "engine_stress", "weather", "airspace_risk"];

function setBadge(el, isAnomaly, score) {
  el.className = `badge ${isAnomaly ? "warn" : "ok"}`;
  el.textContent = `${isAnomaly ? "Anomaly" : "Normal"} (${(score * 100).toFixed(1)}%)`;
}

function metricBox(label, value) {
  return `<div class="metric"><strong>${label}</strong><br>${value}</div>`;
}

function drawMap(flight) {
  const planned = flight.planned_route.map((p) => [p.lat, p.lon]);
  const actual = flight.actual_path.map((p) => [p.lat, p.lon]);

  if (!plannedLine) {
    plannedLine = L.polyline(planned, { color: "#0b7285", weight: 4, dashArray: "8 6" }).addTo(map);
    map.fitBounds(plannedLine.getBounds(), { padding: [20, 20] });
  }

  if (!actualLine) {
    actualLine = L.polyline(actual, { color: "#e76f51", weight: 4 }).addTo(map);
  } else {
    actualLine.setLatLngs(actual);
  }

  anomalyMarkers.forEach((m) => map.removeLayer(m));
  anomalyMarkers = flight.actual_path
    .filter((p) => p.anomaly)
    .slice(-20)
    .map((p) =>
      L.circleMarker([p.lat, p.lon], {
        radius: 5,
        color: "#b91c1c",
        fillColor: "#ef4444",
        fillOpacity: 0.85,
      }).addTo(map)
    );

  const c = flight.current_position;
  if (!currentMarker) {
    currentMarker = L.circleMarker([c.lat, c.lon], {
      radius: 6,
      color: "#1f2937",
      fillColor: "#111827",
      fillOpacity: 1,
    }).addTo(map);
  } else {
    currentMarker.setLatLng([c.lat, c.lon]);
  }
}

function render(data) {
  const { flight, engine, airspace, timestamp, controls } = data;

  drawMap(flight);

  setBadge(engineBadge, engine.is_anomaly, engine.anomaly_score || 0);
  setBadge(airspaceBadge, airspace.is_anomaly, airspace.anomaly_score || 0);

  pathMetrics.innerHTML = [
    metricBox("Distance from Plan (km)", flight.path_metrics.distance_from_plan_km),
    metricBox("Heading Error (deg)", flight.path_metrics.heading_error_deg),
    metricBox("Speed (kts)", flight.path_metrics.speed_kts),
    metricBox("Progress", `${(flight.progress * 100).toFixed(1)}%`),
  ].join("");

  engineMetrics.innerHTML = [
    metricBox("Engine Temp (C)", engine.engine_temp_c.toFixed(2)),
    metricBox("Vibration (g)", engine.vibration_g.toFixed(2)),
    metricBox("Oil Pressure (psi)", engine.oil_pressure_psi.toFixed(2)),
    metricBox("Hydraulic (psi)", engine.hydraulic_psi.toFixed(2)),
    metricBox("Fuel Flow (kg/h)", engine.fuel_flow_kg_h.toFixed(2)),
    metricBox("Bus Voltage (V)", engine.avionics_bus_v.toFixed(2)),
    metricBox("ML Score", engine.ml_score.toFixed(3)),
    metricBox("DL Score", engine.ae_score.toFixed(3)),
  ].join("");

  airspaceMetrics.innerHTML = [
    metricBox("Open Corridors", airspace.open_corridors),
    metricBox("Restricted", airspace.restricted_corridors),
    metricBox("Shutdown", airspace.shutdown_corridors),
    metricBox("Anomaly Score", airspace.anomaly_score.toFixed(3)),
  ].join("");

  newsList.innerHTML = airspace.latest_news
    .slice(0, 6)
    .map((item) => `<li><strong>${item.headline}</strong><br>Severity: ${item.severity} | ${new Date(item.time).toLocaleTimeString()}</li>`)
    .join("");

  controlIds.forEach((id) => {
    const el = document.getElementById(id);
    if (el && document.activeElement !== el) {
      el.value = controls[id];
    }
  });

  clock.textContent = `UTC: ${new Date(timestamp).toLocaleTimeString("en-US", { hour12: false, timeZone: "UTC" })}`;
}

async function fetchState() {
  const res = await fetch("/api/state");
  const data = await res.json();
  render(data);
}

async function applyControls() {
  const controls = {};
  controlIds.forEach((id) => {
    controls[id] = parseFloat(document.getElementById(id).value);
  });

  const injections = {
    engine_temp_delta: parseFloat(document.getElementById("engine_temp_delta").value || "0"),
    vibration_delta: parseFloat(document.getElementById("vibration_delta").value || "0"),
    force_airspace_shutdown: document.getElementById("force_airspace_shutdown").checked,
  };

  await fetch("/api/control", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ controls, injections }),
  });

  document.getElementById("force_airspace_shutdown").checked = false;
}

async function resetSimulation() {
  await fetch("/api/reset", { method: "POST" });
}

const applyBtn = document.getElementById("applyBtn");
const resetBtn = document.getElementById("resetBtn");

function animatePress(btn) {
  btn.classList.remove("press");
  // Force reflow to replay animation on consecutive clicks.
  void btn.offsetWidth;
  btn.classList.add("press");
}

applyBtn.addEventListener("click", async () => {
  animatePress(applyBtn);
  await applyControls();
});

resetBtn.addEventListener("click", async () => {
  animatePress(resetBtn);
  await resetSimulation();
});

fetchState();
setInterval(fetchState, 1500);
