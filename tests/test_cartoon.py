import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robot_portrait.generators.base import GeneratorOutput, GeneratorRequest, GeneratorStatus
from robot_portrait.generators.cartoon import CartoonGenerator
from robot_portrait.io import read_image, write_image
from robot_portrait.pipeline import GenerationPipeline, PipelineConfig
from robot_portrait.quality import evaluate_portrait, tolerant_f1
from robot_portrait.web.app import create_app
from robot_portrait.web.jobs import JobStore


def test_missing_model_does_not_invoke_wsl(tmp_path, monkeypatch):
    model = CartoonGenerator(tmp_path, "Ubuntu", "/env/bin/python")
    monkeypatch.setattr(model, "_run", lambda *a: pytest.fail("must not invoke WSL"))
    status = model.status()
    assert not status.available
    assert "GNR_checkpoint.pt" in status.details["missing"]


def test_wsl_failure_is_reported_as_unavailable(tmp_path, monkeypatch):
    model = CartoonGenerator(tmp_path, "Ubuntu", "/env/bin/python")
    for file in ("model.py", "op/__init__.py", "op/fused_act.py", "op/upfirdn2d.py", "GNR_checkpoint.pt"):
        target = model.repo / file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    monkeypatch.setattr(model, "_run", lambda *a: subprocess.CompletedProcess([], 1, b"", b"CUDA unavailable"))
    assert "CUDA unavailable" in model.status().reason


def test_drive_paths_with_spaces_and_unicode_are_not_shell_code():
    from robot_portrait.generators.apdrawinggan_backend import APDrawingGANGenerator
    assert APDrawingGANGenerator._translate_windows_drive_path(r"D:\my project\画像\input.png") == "/mnt/d/my project/画像/input.png"


def test_cartoon_preserves_colour_and_vectorizes_only_its_contours(tmp_path, monkeypatch):
    source = np.full((256, 256, 3), (200, 120, 60), dtype=np.uint8)
    cv2.circle(source, (128, 128), 65, (10, 25, 60), -1)
    input_path = write_image(tmp_path / "input.png", source)
    model = CartoonGenerator(tmp_path, "Ubuntu", "/env/python")
    monkeypatch.setattr(model, "status", lambda: GeneratorStatus(True))
    def infer(*args):
        write_image(tmp_path / "job/cartoon.png", source)
        return subprocess.CompletedProcess([], 0, b"{}", b"")
    monkeypatch.setattr(model, "_run", infer)
    monkeypatch.setattr(model, "_wsl_path", lambda p: str(p))
    pipeline = GenerationPipeline(PipelineConfig(preprocess_size=256, vector={"mode": "centerline"}), generators={"cartoon": model})
    result = pipeline.run(input_path, tmp_path / "job", method="cartoon")
    assert np.array_equal(read_image(result.raster_path), source)
    lines = read_image(result.vector_input_path, cv2.IMREAD_GRAYSCALE)
    assert lines[128, 128] == 255  # filled colour must not become a filled plotter region
    assert np.count_nonzero(lines == 0) > 50
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["files"]["vector_input"] == "vector_input.png"
    assert manifest["evaluation"]["vector_retention"]["edge_f1"] > .8
    assert result.svg_path.is_file()


def test_mismatched_vector_image_is_rejected():
    with pytest.raises(ValueError, match="vector_image"):
        GeneratorOutput(np.zeros((32, 32, 3)), vector_image=np.zeros((16, 16)))


def test_metrics_do_not_reward_empty_drawings():
    ref = np.zeros((256, 256), dtype=bool)
    ref[100, 30:200] = True
    assert tolerant_f1(ref, ref) == 1
    assert tolerant_f1(ref, np.zeros_like(ref)) == 0
    assert tolerant_f1(np.zeros_like(ref), np.zeros_like(ref)) is None
    assert tolerant_f1(ref, np.roll(ref, 10, axis=0)) == 0


@pytest.mark.parametrize("size,available,code", [(512, True, 422), (256, False, 503), (256, True, 201)])
def test_cartoon_api_size_and_availability(tmp_path, size, available, code):
    class Worker:
        def start(self): pass
        def stop(self, **kwargs): pass
        def submit(self, job_id): pass
    app = create_app(
        generator_status_provider=lambda: {"cartoon": {"available": available}},
        supported_methods_provider=lambda: ("cartoon",),
        job_store=JobStore(tmp_path / "jobs"), job_worker=Worker(),
    )
    ok, data = cv2.imencode(".png", np.full((32, 32, 3), 180, dtype=np.uint8))
    assert ok
    with TestClient(app) as client:
        response = client.post("/api/jobs", data={"method": "cartoon", "size": size},
                               files={"image": ("portrait.png", data.tobytes(), "image/png")})
    assert response.status_code == code
