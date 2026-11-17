import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from bayesian.service import create_app


@pytest.fixture()
def client():
    return TestClient(create_app())


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["active_model"] == "bayesian-logistic"
    assert body["stage"] == "ACTIVE"


def test_predict(client):
    response = client.post("/predict", json={"features": [0.1, -0.2, 0.3, 0.0, 0.5, -0.1]})
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["action"] in ("act", "no_act", "abstain")
    assert body["fallback"] is False


def test_predict_wrong_feature_count(client):
    response = client.post("/predict", json={"features": [0.1, 0.2]})
    assert response.status_code == 422


def test_predict_fallback(client):
    response = client.post(
        "/predict",
        json={"features": [0.1, -0.2, 0.3, 0.0, 0.5, -0.1], "simulate_slow": True},
    )
    assert response.status_code == 200
    assert response.json()["fallback"] is True


def test_predict_batch(client):
    response = client.post(
        "/predict/batch",
        json={"rows": [[0.1] * 6, [-0.5] * 6]},
    )
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2


def test_model_endpoint(client):
    response = client.get("/model")
    assert response.status_code == 200
    body = response.json()
    assert body["evaluation"]["accuracy"] > 0.7
    assert "manifest" in body
    assert body["mcmc"]["acceptance_rate"] > 0


def test_registry_endpoint(client):
    response = client.get("/registry")
    assert response.status_code == 200
    assert response.json()["active_version"] == "bayesian-logistic@v1"


def test_audit_endpoint_verified(client):
    response = client.get("/audit")
    assert response.status_code == 200
    assert response.json()["verified"] is True


def test_drift_endpoint(client):
    response = client.get("/drift")
    assert response.status_code == 200
    body = response.json()
    assert body["action"] == "recalibrate_and_refit"
