from __future__ import annotations

import csv
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / "data"


def load_route() -> list[dict]:
    path = DATA_DIR / "flight_route_seed.csv"
    if not path.exists():
        return []
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                }
            )
    return rows


def load_engine_baseline() -> dict:
    path = DATA_DIR / "engine_sensor_baseline.csv"
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    r = rows[0]
    return {
        "engine_temp_c": float(r["engine_temp_c"]),
        "vibration_g": float(r["vibration_g"]),
        "oil_pressure_psi": float(r["oil_pressure_psi"]),
        "hydraulic_psi": float(r["hydraulic_psi"]),
        "fuel_flow_kg_h": float(r["fuel_flow_kg_h"]),
        "avionics_bus_v": float(r["avionics_bus_v"]),
    }


def load_airspace_news() -> list[dict]:
    path = DATA_DIR / "airspace_news_seed.csv"
    if not path.exists():
        return []
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({"headline": row["headline"], "severity": float(row["severity"])})
    return rows
