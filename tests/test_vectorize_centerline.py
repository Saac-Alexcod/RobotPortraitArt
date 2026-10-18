from __future__ import annotations

from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


from robot_portrait.vectorize import (
    _opencv_svg_to_raster,
    raster_to_svg,
)


def _canvas(
    width: int = 96,
    height: int = 96,
) -> np.ndarray:
    return np.full(
        (height, width, 3),
        255,
        dtype=np.uint8,
    )


def _path_data(
    svg_path: Path,
) -> list[str]:
    root = ET.parse(
        svg_path
    ).getroot()

    return [
        element.attrib["d"]
        for element in root.iter()
        if element.tag.rsplit(
            "}",
            1,
        )[-1] == "path"
    ]


def test_default_contour_mode_keeps_closed_paths_and_motion_metadata(
    tmp_path,
):
    image = _canvas()
    cv2.rectangle(
        image,
        (15, 15),
        (80, 80),
        (0, 0, 0),
        3,
    )

    output = tmp_path / "contour.svg"
    info = raster_to_svg(
        image,
        output,
        min_area=1,
    )

    paths = _path_data(output)

    assert info["mode"] == "contour"
    assert info["path_count"] > 0
    assert info["stroke_count"] == info["path_count"]
    assert info["closed_stroke_count"] == info["path_count"]
    assert info["pen_down_length_px"] > 0
    assert all(
        path.rstrip().endswith("Z")
        for path in paths
    )


def test_centerline_mode_turns_thick_bar_into_one_open_svg_path(
    tmp_path,
):
    image = _canvas()
    cv2.line(
        image,
        (12, 48),
        (83, 48),
        (0, 0, 0),
        9,
        cv2.LINE_8,
    )

    output = tmp_path / "centerline.svg"
    info = raster_to_svg(
        image,
        output,
        mode="centerline",
        centerline_spur_length_px=0,
        centerline_min_stroke_length_px=5,
        centerline_simplify_epsilon_px=0.5,
    )

    paths = _path_data(output)

    assert info["mode"] == "centerline"
    assert info["path_count"] == 1
    assert info["stroke_count"] == 1
    assert info["closed_stroke_count"] == 0
    assert info["skeleton_pixels"] > 60
    assert info["pen_down_length_px"] > 60
    assert info["pen_up_length_px"] == pytest.approx(0.0)
    assert len(paths) == 1
    assert not paths[0].rstrip().endswith("Z")


def test_centerline_mode_preserves_a_closed_loop(
    tmp_path,
):
    image = _canvas()
    diamond = np.asarray(
        [
            [48, 12],
            [84, 48],
            [48, 84],
            [12, 48],
        ],
        dtype=np.int32,
    )

    cv2.polylines(
        image,
        [diamond],
        True,
        (0, 0, 0),
        1,
        cv2.LINE_8,
    )

    output = tmp_path / "loop.svg"
    info = raster_to_svg(
        image,
        output,
        mode="centerline",
        centerline_spur_length_px=0,
        centerline_min_stroke_length_px=1,
        centerline_simplify_epsilon_px=0,
    )

    paths = _path_data(output)

    assert info["stroke_count"] == 1
    assert info["closed_stroke_count"] == 1
    assert len(paths) == 1
    assert paths[0].rstrip().endswith("Z")


def test_opencv_fallback_does_not_close_open_centerline_path(
    tmp_path,
):
    svg = tmp_path / "open.svg"
    png = tmp_path / "open.png"

    svg.write_text(
        """<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 32 32">
<rect width="100%" height="100%" fill="white"/>
<path d="M 5 5 L 25 5 L 25 25" fill="none" stroke="black" stroke-width="1"/>
</svg>
""",
        encoding="utf-8",
    )

    info = _opencv_svg_to_raster(
        svg,
        png,
        32,
    )
    image = cv2.imread(
        str(png),
        cv2.IMREAD_GRAYSCALE,
    )

    assert info["paths"] == 1
    assert info["open_paths"] == 1
    assert info["closed_paths"] == 0
    assert image[15, 15] > 240


def test_opencv_fallback_honors_z_for_closed_path(
    tmp_path,
):
    svg = tmp_path / "closed.svg"
    png = tmp_path / "closed.png"

    svg.write_text(
        """<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 32 32">
<rect width="100%" height="100%" fill="white"/>
<path d="M 5 5 L 25 5 L 25 25 Z" fill="none" stroke="black" stroke-width="1"/>
</svg>
""",
        encoding="utf-8",
    )

    info = _opencv_svg_to_raster(
        svg,
        png,
        32,
    )
    image = cv2.imread(
        str(png),
        cv2.IMREAD_GRAYSCALE,
    )

    assert info["paths"] == 1
    assert info["open_paths"] == 0
    assert info["closed_paths"] == 1
    assert image[15, 15] < 64


def test_empty_centerline_svg_still_rasterizes_to_white_canvas(
    tmp_path,
):
    image = _canvas()
    svg = tmp_path / "empty.svg"
    png = tmp_path / "empty.png"

    vector = raster_to_svg(
        image,
        svg,
        mode="centerline",
    )
    raster = _opencv_svg_to_raster(
        svg,
        png,
        96,
    )
    rendered = cv2.imread(
        str(png),
        cv2.IMREAD_GRAYSCALE,
    )

    assert vector["stroke_count"] == 0
    assert raster["paths"] == 0
    assert rendered.shape == (96, 96)
    assert int(rendered.min()) == 255


def test_invalid_vector_mode_is_rejected(
    tmp_path,
):
    image = _canvas()

    with pytest.raises(
        ValueError,
        match="vector mode",
    ):
        raster_to_svg(
            image,
            tmp_path / "bad.svg",
            mode="not-a-mode",
        )
