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
