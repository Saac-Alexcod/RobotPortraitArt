import json
import sys
from pathlib import Path
import time

import cv2
from fastapi.testclient import TestClient
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


from robot_portrait.io import read_image
from robot_portrait.web.app import create_app
from robot_portrait.web.jobs import JobStore
from robot_portrait.web.runtime import build_default_pipeline
from robot_portrait.web import runtime as web_runtime


class _NoopWorker:
    def __init__(self):
        self.submitted: list[str] = []
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def submit(self, job_id: str):
        self.submitted.append(
            job_id
        )
        return None

    def stop(
        self,
        *,
        wait: bool = True,
    ) -> None:
        self.stopped = True


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


def _test_app(
    tmp_path: Path,
    **kwargs,
):
    worker = _NoopWorker()
    app = create_app(
        supported_methods_provider=lambda: (
            "apdrawinggan",
            "baseline",
        ),
        job_store=JobStore(
            tmp_path / "jobs"
        ),
        job_worker=worker,
        **kwargs,
    )
    return app, worker


def test_health_endpoint():
    worker = _NoopWorker()
    app = create_app(
        job_worker=worker
    )

    with TestClient(app) as client:
        response = client.get(
            "/api/health"
        )
        assert worker.started is True

    assert worker.stopped is True
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

    worker = _NoopWorker()
    with TestClient(
        create_app(
            generator_status_provider=statuses,
            job_worker=worker,
        )
    ) as client:
        response = client.get(
            "/api/methods"
        )

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


def test_methods_endpoint_requires_auto_preprocess_for_web_apdrawinggan():
    def statuses():
        return {
            "apdrawinggan": {
                "available": True,
                "reason": None,
                "details": {
                    "backend": "apdrawinggan",
                    "auto_preprocess": {
                        "available": False,
                        "yunet_model_present": False,
                        "humanseg_model_present": False,
                    },
                },
            },
        }

    worker = _NoopWorker()
    with TestClient(
        create_app(
            generator_status_provider=statuses,
            supported_methods_provider=lambda: (
                "apdrawinggan",
            ),
            job_worker=worker,
        )
    ) as client:
        response = client.get(
            "/api/methods"
        )

    assert response.status_code == 200
    method = response.json()[
        "methods"
    ][0]
    assert method[
        "available"
    ] is False
    assert (
        "YuNet"
        in method[
            "reason"
        ]
    )


def test_create_job_rejects_unavailable_apdrawinggan_before_persisting(
    tmp_path,
):
    def statuses():
        return {
            "apdrawinggan": {
                "available": True,
                "reason": None,
                "details": {
                    "auto_preprocess": {
                        "available": False,
                    },
                },
            },
        }

    store = JobStore(
        tmp_path / "jobs"
    )
    worker = _NoopWorker()
    app = create_app(
        generator_status_provider=statuses,
        supported_methods_provider=lambda: (
            "apdrawinggan",
        ),
        job_store=store,
        job_worker=worker,
    )

    with TestClient(app) as client:
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
                "vector_mode": "centerline",
                "size": "512",
            },
        )

    assert response.status_code == 503
    detail = response.json()[
        "detail"
    ]
    assert detail[
        "code"
    ] == "method_unavailable"
    assert detail[
        "method"
    ] == "apdrawinggan"
    assert worker.submitted == []
    assert not list(
        (tmp_path / "jobs").glob(
            "*/job.json"
        )
    )


def test_create_job_persists_canonical_png_and_metadata(tmp_path):
    app, worker = _test_app(
        tmp_path
    )

    with TestClient(app) as client:
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
    assert worker.submitted == [
        payload["job_id"]
    ]

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
    assert metadata["updated_at"]
    assert metadata["started_at"] is None
    assert metadata["completed_at"] is None
    assert metadata["error"] is None
    assert metadata["result_files"] == {}


def test_create_job_rejects_unsupported_method(tmp_path):
    app, _ = _test_app(
        tmp_path
    )

    with TestClient(app) as client:
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
    app, _ = _test_app(
        tmp_path
    )

    with TestClient(app) as client:
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
    app, _ = _test_app(
        tmp_path
    )

    with TestClient(app) as client:
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
    app, _ = _test_app(
        tmp_path
    )

    with TestClient(app) as client:
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
    app, _ = _test_app(
        tmp_path,
        max_upload_bytes=8,
    )

    with TestClient(app) as client:
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
    worker = _NoopWorker()

    app = create_app(
        supported_methods_provider=lambda: (
            "apdrawinggan",
            "baseline",
        ),
        job_store=JobStore(
            unicode_root / "jobs"
        ),
        job_worker=worker,
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



def test_get_job_returns_persisted_status(tmp_path):
    app, _ = _test_app(
        tmp_path
    )

    with TestClient(app) as client:
        created = client.post(
            "/api/jobs",
            files={
                "image": (
                    "face.png",
                    _png_bytes(),
                    "image/png",
                )
            },
            data={
                "method": "baseline",
                "vector_mode": "centerline",
                "size": "512",
            },
        ).json()

        response = client.get(
            f"/api/jobs/{created['job_id']}"
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"] == created["job_id"]
    assert payload["status"] == "queued"
    assert payload["method"] == "baseline"
    assert payload["vector_mode"] == "centerline"
    assert payload["error"] is None
    assert payload["result_files"] == {}


def test_get_job_returns_404_for_unknown_id(tmp_path):
    app, _ = _test_app(
        tmp_path
    )

    with TestClient(app) as client:
        response = client.get(
            "/api/jobs/"
            "00000000000000000000000000000000"
        )

    assert response.status_code == 404
    assert (
        response.json()["detail"]["code"]
        == "job_not_found"
    )


def test_manifest_returns_conflict_until_job_completed(tmp_path):
    app, _ = _test_app(
        tmp_path
    )

    with TestClient(app) as client:
        created = client.post(
            "/api/jobs",
            files={
                "image": (
                    "face.png",
                    _png_bytes(),
                    "image/png",
                )
            },
        ).json()

        response = client.get(
            f"/api/jobs/{created['job_id']}/manifest"
        )

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "manifest_unavailable"
    assert detail["status"] == "queued"


def test_manifest_returns_completed_pipeline_json(tmp_path):
    store = JobStore(
        tmp_path / "jobs"
    )
    worker = _NoopWorker()
    app = create_app(
        supported_methods_provider=lambda: (
            "baseline",
        ),
        job_store=store,
        job_worker=worker,
    )

    with TestClient(app) as client:
        created = client.post(
            "/api/jobs",
            files={
                "image": (
                    "face.png",
                    _png_bytes(),
                    "image/png",
                )
            },
        ).json()

        job_dir = store.job_dir(
            created["job_id"]
        )
        manifest = {
            "schema_version": 1,
            "status": "completed",
            "method": "baseline",
            "config": {
                "vector": {
                    "mode": "centerline",
                }
            },
        }
        (
            job_dir
            / "manifest.json"
        ).write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        store.mark_completed(
            created["job_id"],
            result_files={
                "manifest": "manifest.json",
            },
        )

        response = client.get(
            f"/api/jobs/{created['job_id']}/manifest"
        )

    assert response.status_code == 200
    assert response.json() == manifest


def test_file_endpoint_serves_only_public_job_files(tmp_path):
    store = JobStore(
        tmp_path / "jobs"
    )
    worker = _NoopWorker()
    app = create_app(
        supported_methods_provider=lambda: (
            "baseline",
        ),
        job_store=store,
        job_worker=worker,
    )

    with TestClient(app) as client:
        created = client.post(
            "/api/jobs",
            files={
                "image": (
                    "face.png",
                    _png_bytes(),
                    "image/png",
                )
            },
        ).json()

        job_dir = store.job_dir(
            created["job_id"]
        )
        svg = (
            job_dir
            / "baseline.svg"
        )
        svg.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg"></svg>',
            encoding="utf-8",
        )
        store.mark_completed(
            created["job_id"],
            result_files={
                "vector_svg": "baseline.svg",
            },
        )

        result = client.get(
            f"/api/jobs/{created['job_id']}"
            "/files/baseline.svg"
        )
        hidden = client.get(
            f"/api/jobs/{created['job_id']}"
            "/files/job.json"
        )

    assert result.status_code == 200
    assert (
        result.headers["content-type"]
        .startswith(
            "image/svg+xml"
        )
    )
    assert b"<svg" in result.content

    assert hidden.status_code == 404
    assert (
        hidden.json()["detail"]["code"]
        == "file_not_available"
    )


def test_file_endpoint_serves_original_input_preview(tmp_path):
    app, _ = _test_app(
        tmp_path
    )

    with TestClient(app) as client:
        created = client.post(
            "/api/jobs",
            files={
                "image": (
                    "face.png",
                    _png_bytes(),
                    "image/png",
                )
            },
        ).json()

        response = client.get(
            f"/api/jobs/{created['job_id']}"
            "/files/input.png"
        )

    assert response.status_code == 200
    assert (
        response.headers["content-type"]
        .startswith(
            "image/png"
        )
    )
    assert response.content


def test_real_worker_results_are_queryable_over_http(tmp_path):
    store = JobStore(
        tmp_path
        / "智能机器人"
        / "jobs"
    )
    app = create_app(
        supported_methods_provider=lambda: (
            "baseline",
        ),
        job_store=store,
    )

    with TestClient(app) as client:
        created = client.post(
            "/api/jobs",
            files={
                "image": (
                    "face.png",
                    _png_bytes(),
                    "image/png",
                )
            },
            data={
                "method": "baseline",
                "vector_mode": "centerline",
                "size": "64",
            },
        ).json()

        deadline = (
            time.monotonic()
            + 20
        )
        status_payload = None

        while time.monotonic() < deadline:
            response = client.get(
                f"/api/jobs/{created['job_id']}"
            )
            assert response.status_code == 200
            status_payload = response.json()

            if status_payload["status"] in {
                "completed",
                "failed",
            }:
                break

            time.sleep(
                0.02
            )

        assert status_payload is not None
        assert status_payload["status"] == "completed", (
            status_payload["error"]
        )

        manifest_response = client.get(
            f"/api/jobs/{created['job_id']}/manifest"
        )
        svg_response = client.get(
            f"/api/jobs/{created['job_id']}"
            "/files/baseline.svg"
        )

    assert manifest_response.status_code == 200
    manifest = manifest_response.json()
    assert manifest["method"] == "baseline"
    assert (
        manifest["config"]["vector"]["mode"]
        == "centerline"
    )

    assert svg_response.status_code == 200
    assert (
        svg_response.headers["content-type"]
        .startswith(
            "image/svg+xml"
        )
    )

def test_openapi_exposes_web_endpoints():
    worker = _NoopWorker()
    with TestClient(
        create_app(
            job_worker=worker
        )
    ) as client:
        schema = client.get(
            "/openapi.json"
        ).json()

    assert "/api/health" in schema["paths"]
    assert "/api/methods" in schema["paths"]
    assert "/api/jobs" in schema["paths"]
    assert "post" in schema["paths"]["/api/jobs"]
    assert "/api/jobs/{job_id}" in schema["paths"]
    assert "/api/jobs/{job_id}/manifest" in schema["paths"]
    assert (
        "/api/jobs/{job_id}/files/{filename}"
        in schema["paths"]
    )


def test_default_statuses_keep_baseline_available_on_preflight_exception(
    monkeypatch,
):
    class _BrokenPipeline:
        def generator_statuses(
            self,
        ):
            raise RuntimeError(
                "synthetic preflight crash"
            )

    monkeypatch.setattr(
        web_runtime,
        "build_default_pipeline",
        lambda: _BrokenPipeline(),
    )

    statuses = (
        web_runtime
        .default_generator_statuses()
    )

    assert statuses[
        "baseline"
    ][
        "available"
    ] is True
    assert statuses[
        "apdrawinggan"
    ][
        "available"
    ] is False
    assert (
        "RuntimeError"
        in statuses[
            "apdrawinggan"
        ][
            "reason"
        ]
    )


def test_default_pipeline_registers_web_generators_without_running_preflight():
    pipeline = build_default_pipeline()

    assert pipeline.supported_methods == (
        "apdrawinggan",
        "baseline",
        "cartoon",
    )


def test_real_web_app_lifespan_marks_previous_queued_job_failed(tmp_path):
    from robot_portrait.web.worker import JobWorker

    store = JobStore(tmp_path / "jobs")
    previous = store.create(
        image_bytes=_png_bytes(),
        original_filename="previous.png",
        method="baseline",
        vector_mode="centerline",
        size=512,
    )
    worker = JobWorker(store, pipeline_builder=lambda **_: None)
    app = create_app(job_store=store, job_worker=worker)
    with TestClient(app) as client:
        response = client.get(f"/api/jobs/{previous.job_id}")
    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert "重启" in response.json()["error"]["message"]


def test_default_preflight_reuses_status_for_methods_and_job_submission(tmp_path, monkeypatch):
    """Expensive WSL preflight is reused, unless explicitly refreshed."""
    from importlib import import_module
    web_app = import_module("robot_portrait.web.app")

    calls = []

    def statuses():
        calls.append(time.monotonic())
        return {
            "apdrawinggan": {
                "available": True,
                "reason": None,
                "details": {"auto_preprocess": {"available": True}},
            },
            "baseline": {"available": True, "reason": None, "details": {}},
        }

    monkeypatch.setattr(web_app, "default_generator_statuses", statuses)
    app, worker = _test_app(tmp_path)
    with TestClient(app) as client:
        assert client.get("/api/methods").status_code == 200
        assert client.get("/api/methods").status_code == 200
        created = client.post(
            "/api/jobs",
            files={"image": ("example.png", _png_bytes(), "image/png")},
            data={"method": "apdrawinggan", "vector_mode": "centerline", "size": "512"},
        )
        assert created.status_code == 201
        assert len(worker.submitted) == 1
        assert len(calls) == 1  # No second costly WSL/CUDA probe.
        assert client.get("/api/methods?refresh=true").status_code == 200
        assert len(calls) == 2
