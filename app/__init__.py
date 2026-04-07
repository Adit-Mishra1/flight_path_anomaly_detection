import atexit
import threading
import time

from flask import Flask, jsonify, render_template, request

from .simulator import FlightSimulationEngine


_simulator = FlightSimulationEngine()
_stop_event = threading.Event()
_worker = None


def _loop() -> None:
    while not _stop_event.is_set():
        _simulator.step()
        time.sleep(1)


def create_app() -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/compare-models")
    def compare_models():
        return render_template("compare_models.html")

    @app.get("/api/state")
    def state():
        return jsonify(_simulator.get_state())

    @app.get("/api/compare")
    def compare_data():
        return jsonify(_simulator.get_model_comparison())

    @app.post("/api/control")
    def control():
        payload = request.get_json(silent=True) or {}
        controls = payload.get("controls", {})
        injections = payload.get("injections", {})
        _simulator.apply_controls(controls, injections)
        return jsonify({"ok": True, "controls": _simulator.controls})

    @app.post("/api/reset")
    def reset():
        _simulator.reset()
        return jsonify({"ok": True})

    _start_worker_once()
    return app


def _start_worker_once() -> None:
    global _worker
    if _worker is not None:
        return
    _worker = threading.Thread(target=_loop, daemon=True)
    _worker.start()


@atexit.register
def _cleanup() -> None:
    _stop_event.set()
