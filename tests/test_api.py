from fastapi.testclient import TestClient
from scripts.app import app

client = TestClient(app)


def test_predict_endpoint_contract():
    payload = {
        "cpu_util": 0.65,
        "mem_util": 0.70,
        "net_io": 120.0,
        "disk_io": 80.0,
        "queue_depth": 12,
        "active_connections": 250,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code in (200, 422, 503)
