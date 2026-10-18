import sys
from pathlib import Path

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


from robot_portrait.web.app import create_app
from robot_portrait.web.runtime import build_default_pipeline


def test_health_endpoint():
    with TestClient(create_app()) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "robot-portrait-api",
    }


def test_methods_endpoint_serializes_available_and_unavailable_methods():
    def statuses():
        return {
            "baseline": {
                "available": True,
                "reason": None,
                "details": {
                    "backend": "opencv",
                },
            },
            "apdrawinggan": {
                "available": False,
                "reason": "CUDA unavailable",
                "details": {
                    "backend": "apdrawinggan",
                    "auto_preprocess": {
                        "available": False,
                    },
                },
            },
        }

    with TestClient(
        create_app(generator_status_provider=statuses)
    ) as client:
        response = client.get("/api/methods")

    assert response.status_code == 200
    assert response.json() == {
        "methods": [
            {
                "name": "apdrawinggan",
                "available": False,
                "reason": "CUDA unavailable",
                "details": {
                    "backend": "apdrawinggan",
                    "auto_preprocess": {
                        "available": False,
                    },
                },
            },
            {
                "name": "baseline",
                "available": True,
                "reason": None,
                "details": {
                    "backend": "opencv",
                },
            },
        ]
    }


def test_openapi_exposes_health_and_methods_endpoints():
    with TestClient(create_app()) as client:
        schema = client.get("/openapi.json").json()

    assert "/api/health" in schema["paths"]
    assert "get" in schema["paths"]["/api/health"]
    assert "/api/methods" in schema["paths"]
    assert "get" in schema["paths"]["/api/methods"]


def test_default_pipeline_registers_web_generators_without_running_preflight():
    pipeline = build_default_pipeline()

    assert pipeline.supported_methods == (
        "apdrawinggan",
        "baseline",
    )
