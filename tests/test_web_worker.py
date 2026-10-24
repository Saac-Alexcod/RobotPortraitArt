import json
import sys
from pathlib import Path
from threading import (
    Event,
    Lock,
)
import time

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


from robot_portrait.web.jobs import JobStore
from robot_portrait.web.runtime import build_pipeline
from robot_portrait.web.worker import JobWorker


def _png_bytes() -> bytes:
    image = np.full(
        (96, 96, 3),
        255,
        dtype=np.uint8,
    )
    cv2.circle(
        image,
        (48, 44),
        28,
        (0, 0, 0),
        2,
    )
    cv2.line(
        image,
        (31, 39),
        (41, 39),
        (0, 0, 0),
        2,
    )
    cv2.line(
        image,
        (55, 39),
        (65, 39),
        (0, 0, 0),
        2,
    )
    cv2.ellipse(
        image,
        (48, 52),
        (12, 8),
        0,
        10,
        170,
        (0, 0, 0),
        2,
    )
    ok, encoded = cv2.imencode(
        ".png",
        image,
    )
    assert ok
    return encoded.tobytes()


def _create_job(
    store: JobStore,
    *,
    vector_mode: str = "contour",
    size: int = 64,
):
    return store.create(
        image_bytes=_png_bytes(),
        original_filename="face.png",
        method="baseline",
        vector_mode=vector_mode,
        size=size,
    )


def _wait_for_status(
    store: JobStore,
    job_id: str,
    terminal: set[str],
    timeout: float = 10.0,
):
    deadline = (
        time.monotonic()
        + timeout
    )
    while time.monotonic() < deadline:
        record = store.get(
            job_id
        )
        if record.status in terminal:
            return record
        time.sleep(0.01)

    raise AssertionError(
        f"job {job_id} 未在 {timeout}s 内进入 {terminal}"
    )


class _BlockingPipeline:
    def __init__(
        self,
        started: Event,
        release: Event,
    ):
        self.started = started
        self.release = release

    def run(
        self,
        input_path,
        output_dir,
        *,
        method,
    ):
        self.started.set()
        assert self.release.wait(
            timeout=5
        )

        output_dir = Path(
            output_dir
        )

        class _Result:
            prepared_path = (
                output_dir
                / "prepared.png"
            )
            raster_path = (
                output_dir
                / f"{method}.png"
            )
            svg_path = (
                output_dir
                / f"{method}.svg"
            )
            vector_render_path = (
                output_dir
                / (
                    f"{method}"
                    "_vector_render.png"
                )
            )
            manifest_path = (
                output_dir
                / "manifest.json"
            )

        for path in (
            _Result.prepared_path,
            _Result.raster_path,
            _Result.svg_path,
            _Result.vector_render_path,
            _Result.manifest_path,
        ):
            path.write_text(
                "ok",
                encoding="utf-8",
            )

        return _Result()


def test_worker_transitions_queued_running_completed(tmp_path):
    store = JobStore(
        tmp_path / "jobs"
    )
    record = _create_job(
        store
    )
    started = Event()
    release = Event()

    worker = JobWorker(
        store,
        pipeline_builder=lambda **kwargs: (
            _BlockingPipeline(
                started,
                release,
            )
        ),
    )

    worker.start()
    try:
        worker.submit(
            record.job_id
        )

        assert started.wait(
            timeout=5
        )
        running = store.get(
            record.job_id
        )
        assert running.status == "running"
        assert running.started_at is not None
        assert running.completed_at is None

        release.set()

        completed = _wait_for_status(
            store,
            record.job_id,
            {"completed"},
        )
    finally:
        release.set()
        worker.stop()

    assert completed.status == "completed"
    assert completed.completed_at is not None
    assert completed.error is None
    assert completed.result_files[
        "manifest"
    ] == "manifest.json"


def test_worker_records_pipeline_failure(tmp_path):
    store = JobStore(
        tmp_path / "jobs"
    )
    record = _create_job(
        store
    )

    def fail_builder(**kwargs):
        raise RuntimeError(
            "synthetic pipeline failure"
        )

    worker = JobWorker(
        store,
        pipeline_builder=fail_builder,
    )
    worker.start()

    try:
        worker.submit(
            record.job_id
        )
        failed = _wait_for_status(
            store,
            record.job_id,
            {"failed"},
        )
    finally:
        worker.stop()

    assert failed.error == {
        "type": "RuntimeError",
        "message": "synthetic pipeline failure",
    }
    assert failed.completed_at is not None
    assert failed.result_files == {}


def test_worker_executes_jobs_strictly_one_at_a_time(tmp_path):
    store = JobStore(
        tmp_path / "jobs"
    )
    first = _create_job(
        store
    )
    second = _create_job(
        store
    )

    lock = Lock()
    active = 0
    max_active = 0

    class _SerialPipeline:
        def run(
            self,
            input_path,
            output_dir,
            *,
            method,
        ):
            nonlocal active, max_active

            with lock:
                active += 1
                max_active = max(
                    max_active,
                    active,
                )

            time.sleep(
                0.05
            )

            with lock:
                active -= 1

            output_dir = Path(
                output_dir
            )

            class _Result:
                prepared_path = (
                    output_dir
                    / "prepared.png"
                )
                raster_path = (
                    output_dir
                    / f"{method}.png"
                )
                svg_path = (
                    output_dir
                    / f"{method}.svg"
                )
                vector_render_path = (
                    output_dir
                    / (
                        f"{method}"
                        "_vector_render.png"
                    )
                )
                manifest_path = (
                    output_dir
                    / "manifest.json"
                )

            return _Result()

    worker = JobWorker(
        store,
        pipeline_builder=lambda **kwargs: (
            _SerialPipeline()
        ),
    )
    worker.start()

    try:
        worker.submit(
            first.job_id
        )
        worker.submit(
            second.job_id
        )

        first_done = _wait_for_status(
            store,
            first.job_id,
            {"completed"},
        )
        second_done = _wait_for_status(
            store,
            second.job_id,
            {"completed"},
        )
    finally:
        worker.stop()

    assert first_done.status == "completed"
    assert second_done.status == "completed"
    assert max_active == 1


def test_worker_runs_real_baseline_centerline_pipeline(tmp_path):
    store = JobStore(
        tmp_path
        / "智能机器人"
        / "jobs"
    )
    record = _create_job(
        store,
        vector_mode="centerline",
        size=64,
    )

    worker = JobWorker(
        store,
        pipeline_builder=build_pipeline,
    )
    worker.start()

    try:
        worker.submit(
            record.job_id
        )
        completed = _wait_for_status(
            store,
            record.job_id,
            {
                "completed",
                "failed",
            },
            timeout=20,
        )
    finally:
        worker.stop()

    assert completed.status == "completed", (
        completed.error
    )

    job_dir = store.job_dir(
        record.job_id
    )

    for filename in (
        "prepared.png",
        "baseline.png",
        "baseline.svg",
        "baseline_vector_render.png",
        "manifest.json",
    ):
        assert (
            job_dir
            / filename
        ).is_file()

    manifest = json.loads(
        (
            job_dir
            / "manifest.json"
        )
        .read_text(
            encoding="utf-8"
        )
    )

    assert manifest["status"] == "completed"
    assert manifest["method"] == "baseline"
    assert (
        manifest["config"]["vector"]["mode"]
        == "centerline"
    )
    assert completed.result_files == {
        "prepared": "prepared.png",
        "raster_portrait": "baseline.png",
        "vector_svg": "baseline.svg",
        "vector_render": "baseline_vector_render.png",
        "manifest": "manifest.json",
    }


def test_post_job_runs_real_worker_pipeline_end_to_end(tmp_path):
    from fastapi.testclient import TestClient
    from robot_portrait.web.app import create_app

    store = JobStore(
        tmp_path
        / "HTTP中文路径"
        / "jobs"
    )
    worker = JobWorker(
        store,
        pipeline_builder=build_pipeline,
    )
    app = create_app(
        supported_methods_provider=lambda: (
            "apdrawinggan",
            "baseline",
        ),
        job_store=store,
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
                "size": "64",
            },
        )
        assert response.status_code == 201
        payload = response.json()
        assert payload["status"] == "queued"

    completed = store.get(
        payload["job_id"]
    )
    assert completed.status == "completed", completed.error
    assert completed.result_files["manifest"] == "manifest.json"


def test_worker_exposes_apdrawinggan_aligned_input(tmp_path):
    store = JobStore(tmp_path / "jobs")
    record = store.create(
        image_bytes=_png_bytes(),
        original_filename="face.png",
        method="apdrawinggan",
        vector_mode="centerline",
        size=512,
    )

    class _APDrawingGANPipeline:
        def run(self, input_path, output_dir, *, method):
            output_dir = Path(output_dir)
            preprocess_dir = output_dir / "apdrawinggan_preprocess"
            preprocess_dir.mkdir(parents=True, exist_ok=True)
            aligned_path = preprocess_dir / "aligned.png"
            aligned_path.write_bytes(b"aligned-image")

            class _Result:
                prepared_path = output_dir / "prepared.png"
                raster_path = output_dir / f"{method}.png"
                svg_path = output_dir / f"{method}.svg"
                vector_render_path = (
                    output_dir / f"{method}_vector_render.png"
                )
                manifest_path = output_dir / "manifest.json"
                generation = {
                    "metadata": {
                        "apdrawinggan_preprocess": {
                            "mode": "auto",
                            "files": {
                                "aligned_image": str(aligned_path),
                            },
                        },
                    },
                }

            for path in (
                _Result.prepared_path,
                _Result.raster_path,
                _Result.svg_path,
                _Result.vector_render_path,
                _Result.manifest_path,
            ):
                path.write_text("ok", encoding="utf-8")

            return _Result()

    worker = JobWorker(
        store,
        pipeline_builder=lambda **kwargs: _APDrawingGANPipeline(),
    )
    worker.start()

    try:
        worker.submit(record.job_id)
        completed = _wait_for_status(
            store,
            record.job_id,
            {"completed", "failed"},
        )
    finally:
        worker.stop()

    assert completed.status == "completed", completed.error
    assert (
        completed.result_files["aligned_input"]
        == "apdrawinggan_aligned.png"
    )

    public_path = store.public_file_path(
        record.job_id,
        "apdrawinggan_aligned.png",
    )
    assert public_path.read_bytes() == b"aligned-image"


def test_worker_start_recovers_orphaned_jobs_without_touching_completed(tmp_path):
    """Unclean shutdown must not leave jobs permanently queued/running."""
    store = JobStore(tmp_path / "jobs")
    data = _png_bytes()
    orphan_queued = store.create(
        image_bytes=data, original_filename="queued.png",
        method="baseline", vector_mode="centerline", size=512,
    )
    orphan_running = store.create(
        image_bytes=data, original_filename="running.png",
        method="baseline", vector_mode="centerline", size=512,
    )
    store.mark_running(orphan_running.job_id)
    completed = store.create(
        image_bytes=data, original_filename="done.png",
        method="baseline", vector_mode="centerline", size=512,
    )
    store.mark_completed(completed.job_id, result_files={"prepared": "prepared.png"})
    recovered = store.fail_interrupted_jobs()
    assert recovered == 2
    for job_id in (orphan_queued.job_id, orphan_running.job_id):
        current = store.get(job_id)
        assert current.status == "failed"
        assert "重启" in current.error["message"]
    assert store.get(completed.job_id).status == "completed"
