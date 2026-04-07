from __future__ import annotations

import math
import random
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from .data_loader import load_airspace_news, load_engine_baseline, load_route
from .models import OnlineBinaryAnomalyModel, TinyAutoencoder

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

    SKLEARN_ENSEMBLES_AVAILABLE = True
except Exception:
    SKLEARN_ENSEMBLES_AVAILABLE = False

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


@dataclass
class Waypoint:
    lat: float
    lon: float


class FlightSimulationEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rng = random.Random(7)

        self.controls = {
            "route_deviation": 0.1,
            "engine_stress": 0.2,
            "weather": 0.1,
            "airspace_risk": 0.2,
        }
        self._injections = {
            "engine_temp_delta": 0.0,
            "vibration_delta": 0.0,
            "force_airspace_shutdown": False,
        }

        self.route = self._build_route()
        self.route_idx = 0
        self.position = Waypoint(self.route[0].lat, self.route[0].lon)

        self.path_model = OnlineBinaryAnomalyModel(n_features=5, threshold=0.65)
        self.engine_model = OnlineBinaryAnomalyModel(n_features=8, threshold=0.7)
        self.airspace_model = OnlineBinaryAnomalyModel(n_features=5, threshold=0.6)
        self.engine_autoencoder = TinyAutoencoder(input_dim=6, hidden_dim=4, lr=0.001)

        self.engine_status = {}
        self.airspace_state = {}
        self.path_state = {}
        self.history = []
        self.compare_history = {}
        self.compare_records = []
        self.static_models = {}
        self.xgb_backend = "xgboost" if XGBOOST_AVAILABLE else "gradient_boosting_fallback"
        self.step_count = 0
        self.reset()

    def reset(self) -> None:
        with self._lock:
            # Reload route each reset so CSV changes reflect without full server restart.
            self.route = self._build_route()
            self.route_idx = 0
            self.position = Waypoint(self.route[0].lat, self.route[0].lon)
            self.history = [{"lat": self.position.lat, "lon": self.position.lon, "anomaly": False}]
            self.compare_history = {
                "live_learning": [{"lat": self.position.lat, "lon": self.position.lon, "anomaly": False}],
                "random_forest": [{"lat": self.position.lat, "lon": self.position.lon, "anomaly": False}],
                "xgboost": [{"lat": self.position.lat, "lon": self.position.lon, "anomaly": False}],
            }
            self.compare_records = []
            self.static_models = self._build_static_baseline_models()
            self.step_count = 0
            self.engine_status = {
                "engine_temp_c": 690.0,
                "vibration_g": 1.8,
                "oil_pressure_psi": 71.0,
                "hydraulic_psi": 3050.0,
                "fuel_flow_kg_h": 2350.0,
                "avionics_bus_v": 28.0,
            }
            baseline = load_engine_baseline()
            if baseline:
                self.engine_status.update(baseline)

            seed_news = load_airspace_news()
            self.airspace_state = {
                "open_corridors": 4,
                "restricted_corridors": 0,
                "shutdown_corridors": 0,
                "latest_news": [
                    {
                        "headline": "No active airspace disruptions in planned corridor.",
                        "severity": 0.05,
                        "time": self._iso_now(),
                    }
                ],
            }
            if seed_news:
                self.airspace_state["latest_news"] = [
                    {"headline": n["headline"], "severity": n["severity"], "time": self._iso_now()}
                    for n in seed_news[:5]
                ]
            self.path_state = {
                "distance_from_plan_km": 0.0,
                "heading_error_deg": 0.0,
                "speed_kts": 465.0,
            }

    def apply_controls(self, controls: dict, injections: dict) -> None:
        with self._lock:
            for k in self.controls:
                if k in controls:
                    self.controls[k] = float(np.clip(controls[k], 0.0, 1.0))
            for k in self._injections:
                if k in injections:
                    if isinstance(self._injections[k], bool):
                        self._injections[k] = bool(injections[k])
                    else:
                        self._injections[k] = float(injections[k])

    def step(self) -> None:
        with self._lock:
            self.step_count += 1
            target_idx = min(self.route_idx + 1, len(self.route) - 1)
            target = self.route[target_idx]

            deviation = self.controls["route_deviation"]
            weather = self.controls["weather"]
            stress = self.controls["engine_stress"]
            air_risk = self.controls["airspace_risk"]

            lat_step = (target.lat - self.position.lat) * 0.25
            lon_step = (target.lon - self.position.lon) * 0.25

            noise_scale = 0.02 + 0.2 * deviation + 0.1 * weather
            self.position.lat += lat_step + self._rng.uniform(-noise_scale, noise_scale)
            self.position.lon += lon_step + self._rng.uniform(-noise_scale, noise_scale)

            if abs(self.position.lat - target.lat) + abs(self.position.lon - target.lon) < 0.15:
                self.route_idx = target_idx

            ref = self.route[self.route_idx]
            distance_km = self._geo_distance_km(self.position, ref)
            heading_error = 180.0 * min(1.0, distance_km / 40.0)
            speed = 470.0 + self._rng.uniform(-8, 8) - 35.0 * weather - 20.0 * deviation

            self.path_state = {
                "distance_from_plan_km": round(distance_km, 3),
                "heading_error_deg": round(heading_error, 2),
                "speed_kts": round(speed, 2),
            }

            # Engine telemetry synthesis.
            self.engine_status["engine_temp_c"] = 685 + 130 * stress + 40 * weather + self._rng.uniform(-8, 8)
            self.engine_status["engine_temp_c"] += self._injections["engine_temp_delta"]

            self.engine_status["vibration_g"] = 1.7 + 2.6 * stress + 0.8 * deviation + self._rng.uniform(-0.15, 0.15)
            self.engine_status["vibration_g"] += self._injections["vibration_delta"]

            self.engine_status["oil_pressure_psi"] = 73 - 22 * stress - 12 * weather + self._rng.uniform(-2.5, 2.5)
            self.engine_status["hydraulic_psi"] = 3050 - 210 * stress - 110 * weather + self._rng.uniform(-30, 30)
            self.engine_status["fuel_flow_kg_h"] = 2300 + 320 * stress + 170 * weather + self._rng.uniform(-35, 35)
            self.engine_status["avionics_bus_v"] = 28.2 - 1.0 * stress + self._rng.uniform(-0.3, 0.3)

            self._update_airspace(air_risk, weather)

            path_x = np.array([
                distance_km,
                heading_error,
                speed,
                weather,
                deviation,
            ], dtype=float)
            path_label = int(distance_km > 7 or heading_error > 20)
            path_score = self.path_model.update(path_x, path_label)
            rf_score = self._static_model_score("random_forest", path_x)
            xgb_score = self._static_model_score("xgboost", path_x)

            engine_x = np.array([
                self.engine_status["engine_temp_c"],
                self.engine_status["vibration_g"],
                self.engine_status["oil_pressure_psi"],
                self.engine_status["hydraulic_psi"],
                self.engine_status["fuel_flow_kg_h"],
                self.engine_status["avionics_bus_v"],
                stress,
                weather,
            ], dtype=float)
            engine_label = int(
                self.engine_status["engine_temp_c"] > 780
                or self.engine_status["vibration_g"] > 3.9
                or self.engine_status["oil_pressure_psi"] < 42
                or self.engine_status["hydraulic_psi"] < 2700
            )
            engine_score_ml = self.engine_model.update(engine_x, engine_label)
            engine_ae_score = self.engine_autoencoder.train_and_score(engine_x[:6])
            engine_score = 0.65 * engine_score_ml + 0.35 * engine_ae_score

            air_features = self._airspace_features(air_risk)
            air_label = int(self.airspace_state["shutdown_corridors"] > 0 or air_features[2] > 0.5)
            air_score = self.airspace_model.update(np.array(air_features, dtype=float), air_label)

            path_anomaly = path_score >= self.path_model.threshold
            rf_anomaly = rf_score >= self.path_model.threshold
            xgb_anomaly = xgb_score >= self.path_model.threshold
            engine_anomaly = engine_score >= self.engine_model.threshold
            air_anomaly = air_score >= self.airspace_model.threshold

            point = {"lat": round(self.position.lat, 5), "lon": round(self.position.lon, 5)}
            self.history.append({**point, "anomaly": bool(path_anomaly)})
            self.history = self.history[-300:]
            self.compare_history["live_learning"].append({**point, "anomaly": bool(path_anomaly)})
            self.compare_history["random_forest"].append({**point, "anomaly": bool(rf_anomaly)})
            self.compare_history["xgboost"].append({**point, "anomaly": bool(xgb_anomaly)})
            self.compare_history["live_learning"] = self.compare_history["live_learning"][-300:]
            self.compare_history["random_forest"] = self.compare_history["random_forest"][-300:]
            self.compare_history["xgboost"] = self.compare_history["xgboost"][-300:]
            self.compare_records.append(
                {
                    "t": self.step_count,
                    "label": int(path_label),
                    "live_score": float(path_score),
                    "rf_score": float(rf_score),
                    "xgb_score": float(xgb_score),
                    "live_pred": int(path_anomaly),
                    "rf_pred": int(rf_anomaly),
                    "xgb_pred": int(xgb_anomaly),
                }
            )
            self.compare_records = self.compare_records[-300:]

            self.path_state["anomaly_score"] = round(float(path_score), 4)
            self.path_state["is_anomaly"] = bool(path_anomaly)

            self.engine_status["anomaly_score"] = round(float(engine_score), 4)
            self.engine_status["ml_score"] = round(float(engine_score_ml), 4)
            self.engine_status["ae_score"] = round(float(engine_ae_score), 4)
            self.engine_status["is_anomaly"] = bool(engine_anomaly)

            self.airspace_state["anomaly_score"] = round(float(air_score), 4)
            self.airspace_state["is_anomaly"] = bool(air_anomaly)

            self._injections["engine_temp_delta"] *= 0.85
            self._injections["vibration_delta"] *= 0.85
            self._injections["force_airspace_shutdown"] = False

    def get_state(self) -> dict:
        with self._lock:
            return {
                "timestamp": self._iso_now(),
                "controls": dict(self.controls),
                "flight": {
                    "progress": round(self.route_idx / max(len(self.route) - 1, 1), 3),
                    "current_position": {
                        "lat": round(self.position.lat, 5),
                        "lon": round(self.position.lon, 5),
                    },
                    "planned_route": [{"lat": p.lat, "lon": p.lon} for p in self.route],
                    "actual_path": list(self.history),
                    "path_metrics": dict(self.path_state),
                },
                "engine": dict(self.engine_status),
                "airspace": dict(self.airspace_state),
            }

    def get_model_comparison(self) -> dict:
        with self._lock:
            live_series = self._model_series("live_learning", "live_score", "live_pred")
            rf_series = self._model_series("random_forest", "rf_score", "rf_pred")
            xgb_series = self._model_series("xgboost", "xgb_score", "xgb_pred")

            live_metrics = self._classification_metrics(
                [x["live_pred"] for x in self.compare_records],
                [x["label"] for x in self.compare_records],
            )
            rf_metrics = self._classification_metrics(
                [x["rf_pred"] for x in self.compare_records],
                [x["label"] for x in self.compare_records],
            )
            xgb_metrics = self._classification_metrics(
                [x["xgb_pred"] for x in self.compare_records],
                [x["label"] for x in self.compare_records],
            )

            return {
                "timestamp": self._iso_now(),
                "flight": {
                    "planned_route": [{"lat": p.lat, "lon": p.lon} for p in self.route],
                    "current_position": {
                        "lat": round(self.position.lat, 5),
                        "lon": round(self.position.lon, 5),
                    },
                    "progress": round(self.route_idx / max(len(self.route) - 1, 1), 3),
                },
                "models": {
                    "live_learning": {
                        "name": "Live Learning",
                        "map_path": list(self.compare_history["live_learning"]),
                        "series": live_series,
                        "summary": live_metrics,
                    },
                    "random_forest": {
                        "name": "Static Random Forest",
                        "map_path": list(self.compare_history["random_forest"]),
                        "series": rf_series,
                        "summary": rf_metrics,
                    },
                    "xgboost": {
                        "name": "Static XGBoost",
                        "backend": self.xgb_backend,
                        "map_path": list(self.compare_history["xgboost"]),
                        "series": xgb_series,
                        "summary": xgb_metrics,
                    },
                },
                "advantage_gap": {
                    "accuracy_live_vs_rf": round(live_metrics["accuracy"] - rf_metrics["accuracy"], 4),
                    "accuracy_live_vs_xgb": round(live_metrics["accuracy"] - xgb_metrics["accuracy"], 4),
                    "f1_live_vs_rf": round(live_metrics["f1"] - rf_metrics["f1"], 4),
                    "f1_live_vs_xgb": round(live_metrics["f1"] - xgb_metrics["f1"], 4),
                },
            }

    def _model_series(self, model_key: str, score_key: str, pred_key: str) -> list[dict]:
        labels = [x["label"] for x in self.compare_records]
        preds = [x[pred_key] for x in self.compare_records]
        running_correct = 0
        series = []
        for idx, row in enumerate(self.compare_records):
            if row[pred_key] == row["label"]:
                running_correct += 1
            precision, recall, f1 = self._prf(preds[: idx + 1], labels[: idx + 1])
            series.append(
                {
                    "t": row["t"],
                    "score": round(float(row[score_key]), 4),
                    "label": int(row["label"]),
                    "pred": int(row[pred_key]),
                    "accuracy": round(running_correct / (idx + 1), 4),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )
        return series

    def _build_static_baseline_models(self) -> dict:
        if not SKLEARN_ENSEMBLES_AVAILABLE:
            return {"random_forest": None, "xgboost": None}

        x_train, y_train = self._generate_training_samples(n_samples=1200)

        rf = RandomForestClassifier(
            n_estimators=120,
            max_depth=7,
            min_samples_leaf=2,
            random_state=42,
        )
        rf.fit(x_train, y_train)

        if XGBOOST_AVAILABLE:
            xgb = XGBClassifier(
                n_estimators=120,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=42,
                eval_metric="logloss",
            )
        else:
            xgb = GradientBoostingClassifier(n_estimators=180, learning_rate=0.06, max_depth=3, random_state=42)
        xgb.fit(x_train, y_train)

        return {"random_forest": rf, "xgboost": xgb}

    def _generate_training_samples(self, n_samples: int = 1000) -> tuple[np.ndarray, np.ndarray]:
        rows = []
        labels = []
        for _ in range(n_samples):
            deviation = self._rng.uniform(0.0, 1.0)
            weather = self._rng.uniform(0.0, 1.0)
            distance = max(0.0, self._rng.gauss(2.0 + 9.0 * deviation + 7.0 * weather, 2.8))
            heading = min(180.0, 1.8 * distance + self._rng.uniform(-6.0, 6.0))
            speed = 470.0 + self._rng.uniform(-12.0, 12.0) - 35.0 * weather - 22.0 * deviation
            label = int(distance > 7.0 or heading > 20.0)

            rows.append([distance, heading, speed, weather, deviation])
            labels.append(label)

        return np.array(rows, dtype=float), np.array(labels, dtype=int)

    def _static_model_score(self, model_key: str, x: np.ndarray) -> float:
        model = self.static_models.get(model_key)
        if model is None:
            return self._heuristic_static_score(x)

        try:
            proba = model.predict_proba(x.reshape(1, -1))
            return float(proba[0][1])
        except Exception:
            pred = int(model.predict(x.reshape(1, -1))[0])
            return float(0.78 if pred else 0.22)

    @staticmethod
    def _heuristic_static_score(x: np.ndarray) -> float:
        distance, heading, speed, weather, deviation = x.tolist()
        raw = (
            0.44 * min(distance / 10.0, 1.6)
            + 0.38 * min(heading / 30.0, 1.5)
            + 0.10 * max(0.0, (460.0 - speed) / 55.0)
            + 0.08 * ((weather + deviation) / 2.0)
        )
        return float(max(0.0, min(1.0, raw)))

    @staticmethod
    def _classification_metrics(preds: list[int], labels: list[int]) -> dict:
        if not labels:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

        correct = sum(int(p == y) for p, y in zip(preds, labels))
        precision, recall, f1 = FlightSimulationEngine._prf(preds, labels)
        return {
            "accuracy": round(correct / len(labels), 4),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    @staticmethod
    def _prf(preds: list[int], labels: list[int]) -> tuple[float, float, float]:
        tp = sum(int(p == 1 and y == 1) for p, y in zip(preds, labels))
        fp = sum(int(p == 1 and y == 0) for p, y in zip(preds, labels))
        fn = sum(int(p == 0 and y == 1) for p, y in zip(preds, labels))

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        return round(precision, 4), round(recall, 4), round(f1, 4)

    def _update_airspace(self, risk: float, weather: float) -> None:
        open_corr = max(1, 4 - int(risk * 2.5))
        restricted = int(risk * 3 + weather * 2 + self._rng.random())
        shutdown = 1 if (self._rng.random() < (0.03 + 0.25 * risk + 0.1 * weather)) else 0

        if self._injections["force_airspace_shutdown"]:
            shutdown = max(shutdown, 1)

        self.airspace_state["open_corridors"] = max(0, open_corr)
        self.airspace_state["restricted_corridors"] = max(0, restricted)
        self.airspace_state["shutdown_corridors"] = shutdown

        if shutdown:
            headline = "Urgent NOTAM: Temporary corridor shutdown due to geopolitical or weather event."
            sev = 0.95
        elif restricted > 1:
            headline = "ATC advisory: Multiple corridors under temporary restrictions."
            sev = 0.65
        else:
            headline = "No severe restriction bulletin in active corridor."
            sev = 0.1 + 0.2 * risk

        news = {
            "headline": headline,
            "severity": round(float(sev), 3),
            "time": self._iso_now(),
        }
        items = self.airspace_state.get("latest_news", [])
        items.insert(0, news)
        self.airspace_state["latest_news"] = items[:10]

    def _airspace_features(self, risk: float) -> list[float]:
        items = self.airspace_state["latest_news"]
        avg_severity = float(np.mean([x["severity"] for x in items])) if items else 0.0
        return [
            float(self.airspace_state["open_corridors"]),
            float(self.airspace_state["restricted_corridors"]),
            float(self.airspace_state["shutdown_corridors"]),
            avg_severity,
            float(risk),
        ]

    def _build_route(self) -> list[Waypoint]:
        loaded = load_route()
        if loaded:
            return [Waypoint(x["lat"], x["lon"]) for x in loaded]
        return [
            Waypoint(37.6213, -122.3790),  # SFO
            Waypoint(38.7, -118.6),
            Waypoint(39.5, -114.9),
            Waypoint(40.2, -110.7),
            Waypoint(40.9, -106.4),
            Waypoint(41.4, -102.2),
            Waypoint(41.8, -98.3),
            Waypoint(41.97, -87.90),  # ORD
        ]

    @staticmethod
    def _geo_distance_km(a: Waypoint, b: Waypoint) -> float:
        dx = (a.lat - b.lat) * 111.0
        dy = (a.lon - b.lon) * 85.0
        return math.sqrt(dx * dx + dy * dy)

    @staticmethod
    def _iso_now() -> str:
        return datetime.now(timezone.utc).isoformat()
