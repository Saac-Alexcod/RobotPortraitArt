import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src"),
)

from robot_portrait.io import (
    read_image,
    write_image,
)

from robot_portrait.preprocessing import (
    load_haar_detector,
)

from robot_portrait.vectorize import (
    _opencv_svg_to_raster,
    raster_to_svg,
)


def test_unicode_image_io_roundtrip(
    tmp_path,
):
    path = (
        tmp_path
        / "测试 图像.png"
    )

    source = np.zeros(
        (24, 32, 3),
        dtype=np.uint8,
    )

    source[:, :, 1] = 127

    write_image(
        path,
        source,
    )

    loaded = read_image(
        path
    )

    assert (
        loaded.shape
        == source.shape
    )

    assert np.array_equal(
        loaded,
        source,
    )


def test_read_image_rejects_corrupt_file(
    tmp_path,
):
    path = (
        tmp_path
        / "broken.png"
    )

    path.write_bytes(
        b"not an image"
    )

    with pytest.raises(
        ValueError,
        match="corrupted",
    ):
        read_image(
            path
        )


def test_write_image_rejects_unknown_extension(
    tmp_path,
):
    image = np.zeros(
        (8, 8, 3),
        dtype=np.uint8,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Unsupported output "
            "image extension"
        ),
    ):
        write_image(
            tmp_path
            / "image.xyz",
            image,
        )


def test_haar_detector_loads_from_unicode_path(
    tmp_path,
):
    source = (
        Path(
            cv2.data.haarcascades
        )
        / "haarcascade_frontalface_default.xml"
    )

    unicode_dir = (
        tmp_path
        / "智能系统"
    )

    unicode_dir.mkdir()

    target = (
        unicode_dir
        / source.name
    )

    target.write_bytes(
        source.read_bytes()
    )

    detector = (
        load_haar_detector(
            target
        )
    )

    assert detector is not None
    assert not detector.empty()


def test_svg_fallback_writes_unicode_path(
    tmp_path,
):
    image = np.full(
        (32, 32),
        255,
        dtype=np.uint8,
    )

    cv2.rectangle(
        image,
        (5, 5),
        (25, 25),
        0,
        2,
    )

    unicode_dir = (
        tmp_path
        / "矢量结果"
    )

    svg_path = (
        unicode_dir
        / "线稿.svg"
    )

    png_path = (
        unicode_dir
        / "回栅格.png"
    )

    raster_to_svg(
        image,
        svg_path,
        min_area=1,
    )

    info = (
        _opencv_svg_to_raster(
            svg_path,
            png_path,
            64,
        )
    )

    assert (
        info["backend"]
        == "opencv-fallback"
    )

    loaded = read_image(
        png_path
    )

    assert (
        loaded.shape[:2]
        == (64, 64)
    )