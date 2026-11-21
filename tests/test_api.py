from fastapi.testclient import TestClient

from scripts.app import app

VALID_PAYLOAD = {
    "cpu_utilization_pct": 57.48,
    "memory_utilization_pct": 48.94,
    "request_rate_rps": 200.15,
    "p99_latency_ms": 32.16,
    "queue_depth": 4,
    "uptime_hours": 1602,
}

FEATURE_ORDER = [
    "cpu_utilization_pct",
    "memory_utilization_pct",
    "request_rate_rps",
    "p99_latency_ms",
    "queue_depth",
    "uptime_hours",
]


def test_health():
    # Le lifespan charge le modèle : utiliser la forme context manager.
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


def test_predict_telemetry_success():
    with TestClient(app) as client:
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 200
        score = resp.json()["saturation_risk_score"]
        assert 0.0 <= score <= 1.0


def test_predict_vector_success():
    features = [VALID_PAYLOAD[f] for f in FEATURE_ORDER]
    with TestClient(app) as client:
        resp = client.post("/predict", json={"features": features})
        assert resp.status_code == 200
        assert 0.0 <= resp.json()["saturation_risk_score"] <= 1.0


def test_predict_invalid_rejected():
    bad = {**VALID_PAYLOAD, "cpu_utilization_pct": 150.0}
    with TestClient(app) as client:
        resp = client.post("/predict", json=bad)
        assert resp.status_code == 422


def test_predict_vector_out_of_range_rejected():
    features = [VALID_PAYLOAD[f] for f in FEATURE_ORDER]
    features[0] = 150.0
    with TestClient(app) as client:
        resp = client.post("/predict", json={"features": features})
        assert resp.status_code == 422


def test_readiness_requires_loaded_model():
    with TestClient(app) as client:
        resp = client.get("/ready")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ready"}


def test_metrics_expose_prediction_activity():
    with TestClient(app) as client:
        before = client.get("/metrics").json()
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 200
        after = client.get("/metrics").json()
        assert after["prediction_count"] == before["prediction_count"] + 1
        assert after["average_latency_seconds"] >= 0.0


def test_drift_endpoint_reports_status():
    with TestClient(app) as client:
        response = client.get("/drift")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] in {"insufficient_samples", "ready"}
        assert len(payload["features"]) == 6
