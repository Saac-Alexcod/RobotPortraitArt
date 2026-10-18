import sys
from pathlib import Path

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


from robot_portrait.web.app import create_app


def test_health_endpoint():
    with TestClient(create_app()) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "robot-portrait-api",
    }


def test_openapi_exposes_health_endpoint():
    with TestClient(create_app()) as client:
        schema = client.get("/openapi.json").json()

    assert "/api/health" in schema["paths"]
    assert "get" in schema["paths"]["/api/health"]
