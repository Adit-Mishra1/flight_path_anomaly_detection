# Flight Anomaly Live Learning Monitor (Flask)

This project provides a live ML/DL-style monitoring dashboard for flight operations from departure to arrival.

## What it includes

- Flask web app with a 3-window interface:
  - Left large panel: map, planned route, actual path, path anomalies
  - Right top panel: engine and subsystem telemetry + anomaly status
  - Right bottom panel: airspace corridor status + latest disruption bulletins
- Live-learning behavior:
  - Online ML model (`SGDClassifier` with `partial_fit`) for path, engine, and airspace anomaly scoring
  - Tiny online autoencoder-like learner for engine reconstruction error (DL-style signal)
  - Combined engine anomaly decision updates continuously as new simulated points arrive
- Utility window for simulation controls:
  - Route deviation, engine stress, weather severity, airspace risk
  - One-shot event injection: temp spike, vibration spike, forced airspace shutdown

## Seed datasets used

- `app/data/flight_route_seed.csv`
- `app/data/engine_sensor_baseline.csv`
- `app/data/airspace_news_seed.csv`

These local CSVs are loaded by the simulator and can be replaced with real datasets.

## Run

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run.py
```

Open: `http://127.0.0.1:5000`

## API endpoints

- `GET /api/state` live dashboard payload
- `POST /api/control` update controls/injections
- `POST /api/reset` reset simulation state
