const MODEL_CONFIGS = [
  { key: "live_learning", mapId: "liveMap", chartId: "liveChart", summaryId: "liveSummary", color: "#0b7285" },
  { key: "random_forest", mapId: "rfMap", chartId: "rfChart", summaryId: "rfSummary", color: "#8b5cf6" },
  { key: "xgboost", mapId: "xgbMap", chartId: "xgbChart", summaryId: "xgbSummary", color: "#f97316" },
];

const maps = {};

function initMap(id) {
  const map = L.map(id, { zoomControl: false, attributionControl: false }).setView([22.5, 78.9], 4);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 10,
    attribution: "&copy; OpenStreetMap contributors",
  }).addTo(map);

  return {
    map,
    plannedLine: null,
    actualLine: null,
    marker: null,
    anomalyMarkers: [],
  };
}

function drawModelMap(state, plannedRoute, modelPath, color) {
  const planned = plannedRoute.map((p) => [p.lat, p.lon]);
  const actual = modelPath.map((p) => [p.lat, p.lon]);

  if (!state.plannedLine) {
    state.plannedLine = L.polyline(planned, { color: "#1f2937", weight: 3, dashArray: "6 6" }).addTo(state.map);
    state.map.fitBounds(state.plannedLine.getBounds(), { padding: [18, 18] });
  }

  if (!state.actualLine) {
    state.actualLine = L.polyline(actual, { color, weight: 4 }).addTo(state.map);
  } else {
    state.actualLine.setLatLngs(actual);
  }

  state.anomalyMarkers.forEach((m) => state.map.removeLayer(m));
  state.anomalyMarkers = modelPath
    .filter((p) => p.anomaly)
    .slice(-30)
    .map((p) =>
      L.circleMarker([p.lat, p.lon], {
        radius: 4,
        color: "#b91c1c",
        fillColor: "#ef4444",
        fillOpacity: 0.9,
      }).addTo(state.map)
    );

  const current = modelPath[modelPath.length - 1];
  if (!current) return;

  if (!state.marker) {
    state.marker = L.circleMarker([current.lat, current.lon], {
      radius: 5,
      color: "#111827",
      fillColor: "#111827",
      fillOpacity: 1,
    }).addTo(state.map);
  } else {
    state.marker.setLatLng([current.lat, current.lon]);
  }
}

function drawAxes(ctx, w, h) {
  ctx.strokeStyle = "#cbd5e1";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(42, 16);
  ctx.lineTo(42, h - 32);
  ctx.lineTo(w - 16, h - 32);
  ctx.stroke();
}

function drawSeries(ctx, data, color, w, h) {
  const minX = 42;
  const maxX = w - 16;
  const minY = 16;
  const maxY = h - 32;

  if (data.length < 2) return;

  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();

  data.forEach((val, idx) => {
    const x = minX + (idx / (data.length - 1)) * (maxX - minX);
    const y = maxY - (val / 100) * (maxY - minY);
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });

  ctx.stroke();
}

function drawChart(canvas, series, color) {
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(320, rect.width * window.devicePixelRatio);
  canvas.height = Math.max(240, rect.height * window.devicePixelRatio);
  ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

  const w = rect.width;
  const h = rect.height;
  ctx.clearRect(0, 0, w, h);
  drawAxes(ctx, w, h);

  const acc = series.map((x) => x.accuracy * 100);
  const f1 = series.map((x) => x.f1 * 100);

  drawSeries(ctx, acc, color, w, h);
  drawSeries(ctx, f1, "#dc2626", w, h);

  ctx.fillStyle = "#334155";
  ctx.font = "12px Segoe UI";
  ctx.fillText("0", 30, h - 29);
  ctx.fillText("100", 18, 20);
  ctx.fillText("Accuracy", 56, 20);
  ctx.fillStyle = "#dc2626";
  ctx.fillText("F1", 120, 20);
}

function fmtPct(val) {
  return `${(val * 100).toFixed(2)}%`;
}

function renderGap(gap) {
  const el = document.getElementById("gapGrid");
  el.innerHTML = [
    ["Live - RF Accuracy", gap.accuracy_live_vs_rf],
    ["Live - XGB Accuracy", gap.accuracy_live_vs_xgb],
    ["Live - RF F1", gap.f1_live_vs_rf],
    ["Live - XGB F1", gap.f1_live_vs_xgb],
  ]
    .map(([label, value]) => `<div class="gap-item"><strong>${label}</strong><span>${fmtPct(value)}</span></div>`)
    .join("");
}

function renderSummary(id, summary) {
  const el = document.getElementById(id);
  el.textContent = `Acc ${fmtPct(summary.accuracy)} | F1 ${fmtPct(summary.f1)}`;
}

function render(data) {
  document.getElementById("compareTimestamp").textContent = `UTC ${new Date(data.timestamp).toLocaleTimeString("en-US", {
    hour12: false,
    timeZone: "UTC",
  })}`;

  renderGap(data.advantage_gap);

  MODEL_CONFIGS.forEach((cfg) => {
    const model = data.models[cfg.key];
    drawModelMap(maps[cfg.key], data.flight.planned_route, model.map_path, cfg.color);
    drawChart(document.getElementById(cfg.chartId), model.series, cfg.color);
    renderSummary(cfg.summaryId, model.summary);
  });

  const xgb = data.models.xgboost;
  const backendEl = document.getElementById("xgbBackend");
  backendEl.textContent =
    xgb.backend === "xgboost"
      ? "Backend: native XGBoost"
      : "Backend: Gradient Boosting fallback (xgboost package unavailable)";
}

async function refresh() {
  const res = await fetch("/api/compare");
  const data = await res.json();
  render(data);
}

MODEL_CONFIGS.forEach((cfg) => {
  maps[cfg.key] = initMap(cfg.mapId);
});

refresh();
setInterval(refresh, 2000);
window.addEventListener("resize", refresh);
