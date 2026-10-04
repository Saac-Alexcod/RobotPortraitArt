import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from robot_portrait.lineart import portrait_lineart
from robot_portrait.metrics import edge_f1, ssim_score
from robot_portrait.vectorize import _opencv_svg_to_raster, raster_to_svg
from prepare_image import prepare_image
from batch_evaluate import evaluate, FIELDS


def test_metrics_identical_image_are_perfect():
    image = np.zeros((32, 32), dtype=np.uint8)
    image[8:24, 8:24] = 255
    assert ssim_score(image, image, size=32) == pytest.approx(1.0)
    assert edge_f1(image, image, size=32) == pytest.approx(1.0)


def test_vectorize_writes_svg(tmp_path):
    image = np.full((32, 32), 255, dtype=np.uint8)
    cv2.rectangle(image, (5, 5), (25, 25), 0, 2)
    output = tmp_path / "line.svg"
    info = raster_to_svg(image, output, min_area=1)
    assert output.exists()
    assert info["path_count"] > 0
    assert '<svg ' in output.read_text(encoding="utf-8")


def test_opencv_svg_fallback(tmp_path):
    image = np.full((32, 32), 255, dtype=np.uint8)
    cv2.rectangle(image, (5, 5), (25, 25), 0, 2)
    svg_path, png_path = tmp_path / "line.svg", tmp_path / "line.png"
    raster_to_svg(image, svg_path, min_area=1)
    info = _opencv_svg_to_raster(svg_path, png_path, 64)
    assert info["backend"] == "opencv-fallback"
    assert cv2.imread(str(png_path)).shape[:2] == (64, 64)


def test_prepare_image_center_fallback(tmp_path):
    source = tmp_path / "input.png"
    output = tmp_path / "prepared.png"
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    image[:, :, 1] = 128
    assert cv2.imwrite(str(source), image)
    result = prepare_image(source, output, 32)
    assert result["mode"] in {"center", "face"}
    assert cv2.imread(str(output)).shape[:2] == (32, 32)


def test_batch_evaluate_writes_required_columns(tmp_path):
    ref_dir, pred_dir = tmp_path / "ref", tmp_path / "pred"
    ref_dir.mkdir()
    pred_dir.mkdir()
    image = np.full((32, 32, 3), 255, dtype=np.uint8)
    cv2.circle(image, (16, 16), 8, (0, 0, 0), 1)
    cv2.imwrite(str(ref_dir / "a.png"), image)
    cv2.imwrite(str(pred_dir / "a.png"), image)
    csv_path, summary_path = tmp_path / "metrics.csv", tmp_path / "summary.json"
    summary = evaluate(ref_dir, [f"baseline={pred_dir}"], csv_path, summary_path)
    assert summary["completed_rows"] == 1
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["sample_id"] == "a"
    assert list(rows[0]) == FIELDS
    assert json.loads(summary_path.read_text(encoding="utf-8"))["rows"] == 1
