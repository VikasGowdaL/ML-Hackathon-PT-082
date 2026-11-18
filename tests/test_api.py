"""Tests for the FastAPI app: health check and two-laptop CORS configuration."""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


def _app(monkeypatch, origins=None):
    if origins is None:
        monkeypatch.delenv("PATHSNAP_CORS_ORIGINS", raising=False)
    else:
        monkeypatch.setenv("PATHSNAP_CORS_ORIGINS", origins)
    import pathsnap.api.main as main
    return importlib.reload(main).app


def _preflight(client, origin):
    return client.options("/api/health", headers={"Origin": origin, "Access-Control-Request-Method": "GET"})


def test_health(monkeypatch):
    r = TestClient(_app(monkeypatch)).get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


@pytest.mark.parametrize("origin", ["http://localhost:5173", "http://192.168.1.25:5173", "http://10.0.0.7:5173",
                                    "http://172.20.3.4:3000"])
def test_default_cors_allows_local_and_private_lan(monkeypatch, origin):
    r = _preflight(TestClient(_app(monkeypatch)), origin)
    assert r.status_code == 200 and r.headers.get("access-control-allow-origin") == origin


@pytest.mark.parametrize("origin", ["http://8.8.8.8:5173", "http://example.com", "http://172.40.0.1:5173"])
def test_default_cors_blocks_public_origins(monkeypatch, origin):
    r = _preflight(TestClient(_app(monkeypatch)), origin)
    assert r.headers.get("access-control-allow-origin") is None


def test_cors_env_override(monkeypatch):
    client = TestClient(_app(monkeypatch, "http://192.168.1.20:5173"))
    assert _preflight(client, "http://192.168.1.20:5173").headers.get("access-control-allow-origin") == "http://192.168.1.20:5173"
    assert _preflight(client, "http://192.168.1.21:5173").headers.get("access-control-allow-origin") is None
    star = TestClient(_app(monkeypatch, "*"))
    assert _preflight(star, "http://anything.example").headers.get("access-control-allow-origin") == "*"
    _app(monkeypatch)  # restore default app for other tests


def _real_ready():
    from pathsnap import pipeline as P
    return "00000052" in P.trips_index() and P.model_path().exists()


def test_eval_trips_and_predict(monkeypatch):
    if not _real_ready():
        pytest.skip("raw data or final model not present")
    monkeypatch.setenv("PATHSNAP_EVAL_PARTITION", "student_test")  # rehearsal mode: no evaluator data needed
    client = TestClient(_app(monkeypatch))
    ev = client.get("/api/eval/trips").json()
    assert ev["partition"] == "student_test" and ev["count"] == 10
    assert any(t["trip_id"] == "00000052" and t["gps_count"] > 0 for t in ev["trips"])

    r = client.post("/api/predict/00000052")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["gps_count"] == len(body["gps_points"]) > 0
    assert body["predicted_route"] and all(len(sec) >= 2 for sec in body["predicted_route"])
    m = body["metrics"]
    assert 0 < m["route_f1"] <= 1 and 0 <= m["points_on_route"] <= 1 and not body["cached"]
    assert client.post("/api/predict/00000052").json()["cached"] is True
    assert client.post("/api/predict/99999999").status_code == 404
    monkeypatch.delenv("PATHSNAP_EVAL_PARTITION")
    _app(monkeypatch)
