import json
import sys
from pathlib import Path

import cv2
from fastapi.testclient import TestClient
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


from robot_portrait.io import read_image
from robot_portrait.web.app import create_app
from robot_portrait.web.jobs import JobStore
from robot_portrait.web.runtime import build_default_pipeline


def _png_bytes() -> bytes:
    image = np.full(
        (32, 32, 3),
        255,
        dtype=np.uint8,
    )
    cv2.circle(
        image,
        (16, 16),
        8,
        (0, 0, 0),
        1,
    )
    ok, encoded = cv2.imencode(
        ".png",
        image,
    )
    assert ok
    return encoded.tobytes()


def _test_app(tmp_path: Path, **kwargs):
    return create_app(
        supported_methods_provider=lambda: (
            "apdrawinggan",
            "baseline",
        ),
        job_store=JobStore(
            tmp_path / "jobs"
        ),
        **kwargs,
    )


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


def test_create_job_persists_canonical_png_and_metadata(tmp_path):
    with TestClient(
        _test_app(tmp_path)
    ) as client:
        response = client.post(
            "/api/jobs",
            files={
                "image": (
                    "portrait.jpg",
                    _png_bytes(),
                    "image/png",
                )
            },
            data={
                "method": "baseline",
                "vector_mode": "centerline",
                "size": "512",
            },
        )

    assert response.status_code == 201
    payload = response.json()
    assert payload["status"] == "queued"
    assert payload["method"] == "baseline"
    assert payload["vector_mode"] == "centerline"
    assert payload["size"] == 512
    assert len(payload["job_id"]) == 32

    job_dir = (
        tmp_path
        / "jobs"
        / payload["job_id"]
    )
    input_path = job_dir / "input.png"
    metadata_path = job_dir / "job.json"

    assert input_path.is_file()
    assert read_image(input_path) is not None
    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )
    assert metadata["job_id"] == payload["job_id"]
    assert metadata["status"] == "queued"
    assert metadata["original_filename"] == "portrait.jpg"
    assert metadata["input_file"] == "input.png"


def test_create_job_rejects_unsupported_method(tmp_path):
    with TestClient(
        _test_app(tmp_path)
    ) as client:
        response = client.post(
            "/api/jobs",
            files={
                "image": (
                    "face.png",
                    _png_bytes(),
                    "image/png",
                )
            },
            data={
                "method": "missing",
            },
        )

    assert response.status_code == 422
    assert (
        response.json()["detail"]["code"]
        == "unsupported_method"
    )


def test_create_job_rejects_invalid_vector_mode(tmp_path):
    with TestClient(
        _test_app(tmp_path)
    ) as client:
        response = client.post(
            "/api/jobs",
            files={
                "image": (
                    "face.png",
                    _png_bytes(),
                    "image/png",
                )
            },
            data={
                "vector_mode": "polygon",
            },
        )

    assert response.status_code == 422
    assert (
        response.json()["detail"]["code"]
        == "unsupported_vector_mode"
    )


def test_create_job_requires_apdrawinggan_size_512(tmp_path):
    with TestClient(
        _test_app(tmp_path)
    ) as client:
        response = client.post(
            "/api/jobs",
            files={
                "image": (
                    "face.png",
                    _png_bytes(),
                    "image/png",
                )
            },
            data={
                "method": "apdrawinggan",
                "size": "256",
            },
        )

    assert response.status_code == 422
    assert (
        response.json()["detail"]["code"]
        == "apdrawinggan_size"
    )


def test_create_job_rejects_invalid_image_bytes(tmp_path):
    with TestClient(
        _test_app(tmp_path)
    ) as client:
        response = client.post(
            "/api/jobs",
            files={
                "image": (
                    "face.png",
                    b"not-an-image",
                    "image/png",
                )
            },
        )

    assert response.status_code == 400
    assert (
        response.json()["detail"]["code"]
        == "invalid_image"
    )


def test_create_job_rejects_oversized_upload(tmp_path):
    with TestClient(
        _test_app(
            tmp_path,
            max_upload_bytes=8,
        )
    ) as client:
        response = client.post(
            "/api/jobs",
            files={
                "image": (
                    "face.png",
                    _png_bytes(),
                    "image/png",
                )
            },
        )

    assert response.status_code == 413
    assert (
        response.json()["detail"]["code"]
        == "upload_too_large"
    )



def test_create_job_supports_unicode_storage_path(tmp_path):
    unicode_root = (
        tmp_path
        / "智能机器人与智能信息系统"
        / "肖像绘制"
    )

    app = create_app(
        supported_methods_provider=lambda: (
            "apdrawinggan",
            "baseline",
        ),
        job_store=JobStore(
            unicode_root / "jobs"
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/jobs",
            files={
                "image": (
                    "人像.png",
                    _png_bytes(),
                    "image/png",
                )
            },
            data={
                "method": "baseline",
                "vector_mode": "centerline",
                "size": "512",
            },
        )

    assert response.status_code == 201

    payload = response.json()
    input_path = (
        unicode_root
        / "jobs"
        / payload["job_id"]
        / "input.png"
    )

    assert input_path.is_file()
    image = read_image(input_path)
    assert image.shape[:2] == (32, 32)

def test_openapi_exposes_web_endpoints():
    with TestClient(create_app()) as client:
        schema = client.get("/openapi.json").json()

    assert "/api/health" in schema["paths"]
    assert "/api/methods" in schema["paths"]
    assert "/api/jobs" in schema["paths"]
    assert "post" in schema["paths"]["/api/jobs"]


def test_default_pipeline_registers_web_generators_without_running_preflight():
    pipeline = build_default_pipeline()

    assert pipeline.supported_methods == (
        "apdrawinggan",
        "baseline",
    )
