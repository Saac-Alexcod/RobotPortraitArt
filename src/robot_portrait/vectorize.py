from __future__ import annotations

from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from .centerline import (
    CenterlineConfig,
    Stroke,
    extract_centerline_strokes,
    order_strokes_greedy,
    pen_up_length,
)
from .io import write_image


_VECTOR_MODES = {
    "contour",
    "centerline",
}


def _stroke_svg_path(
    stroke: Stroke,
) -> str:
    points = stroke.points
    d = [
        f"M {points[0][0]} {points[0][1]}"
    ]

    d.extend(
        f"L {x} {y}"
        for x, y in points[1:]
    )

    if stroke.closed:
        d.append("Z")

    return " ".join(d)


def _svg_document(
    *,
    width: int,
    height: int,
    paths: list[str],
) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        f'<rect width="100%" height="100%" fill="white"/>\n'
        + "\n".join(paths)
        + "\n</svg>\n"
    )


def _path_element(
    stroke: Stroke,
    *,
    stroke_width: float,
) -> str:
    return (
        f'<path d="{_stroke_svg_path(stroke)}" '
        f'fill="none" '
        f'stroke="black" '
        f'stroke-width="{stroke_width}" '
        f'stroke-linejoin="round" '
        f'stroke-linecap="round"/>'
    )


def _write_svg(
    output_svg: str | Path,
    *,
    width: int,
    height: int,
    strokes: tuple[Stroke, ...],
    stroke_width: float,
) -> Path:
    output_svg = Path(output_svg)

    output_svg.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = [
        _path_element(
            stroke,
            stroke_width=stroke_width,
        )
        for stroke in strokes
    ]

    output_svg.write_text(
        _svg_document(
            width=width,
            height=height,
            paths=paths,
        ),
        encoding="utf-8",
    )

    return output_svg


def _contour_strokes(
    line_img: np.ndarray,
    *,
    min_area: float,
    epsilon_ratio: float,
) -> tuple[
    tuple[Stroke, ...],
    int,
    int,
]:
    if line_img.ndim == 3:
        gray = cv2.cvtColor(
            line_img,
            cv2.COLOR_BGR2GRAY,
        )
    else:
        gray = line_img

    if (
        gray.ndim != 2
        or gray.dtype != np.uint8
        or gray.size == 0
    ):
        raise ValueError(
            "line_img 必须是非空 uint8 灰度图或 BGR 图像"
        )

    # Convert dark strokes into white foreground for contour tracing.
    fg = 255 - gray

    _, fg = cv2.threshold(
        fg,
        127,
        255,
        cv2.THRESH_BINARY,
    )

    contours, _ = cv2.findContours(
        fg,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE,
    )

    strokes: list[Stroke] = []

    for cnt in contours:
        area = abs(
            cv2.contourArea(cnt)
        )

        if area < float(min_area):
            continue

        peri = cv2.arcLength(
            cnt,
            True,
        )

        eps = max(
            0.5,
            float(epsilon_ratio) * peri,
        )

        approx = cv2.approxPolyDP(
            cnt,
            eps,
            True,
        ).reshape(-1, 2)

        if len(approx) < 2:
            continue

        points = tuple(
            (
                int(x),
                int(y),
            )
            for x, y in approx.tolist()
        )

        strokes.append(
            Stroke(
                points=points,
                closed=True,
            )
        )

    height, width = gray.shape[:2]

    return (
        tuple(strokes),
        width,
        height,
    )


def _motion_metadata(
    strokes: tuple[Stroke, ...],
) -> dict[str, object]:
    return {
        "stroke_count": len(strokes),
        "pen_down_length_px": float(
            sum(
                stroke.length_px
                for stroke in strokes
            )
        ),
        "pen_up_length_px": float(
            pen_up_length(
                strokes
            )
        ),
        "closed_stroke_count": sum(
            1
            for stroke in strokes
            if stroke.closed
        ),
    }


def raster_to_svg(
    line_img: np.ndarray,
    output_svg: str | Path,
    min_area: float = 12.0,
    epsilon_ratio: float = 0.0025,
    stroke_width: float = 1.2,
    mode: str = "contour",
    centerline_threshold: int = 220,
    centerline_spur_length_px: float = 3.0,
    centerline_min_stroke_length_px: float = 2.0,
    centerline_simplify_epsilon_px: float = 0.8,
    centerline_max_thinning_iterations: int = 2048,
    centerline_junction_cluster_radius_px: int = 1,
    centerline_merge_angle_deg: float = 35.0,
    centerline_merge_lookahead_px: float = 6.0,
    centerline_ordering_2opt_max_passes: int = 16,
    centerline_ordering_2opt_min_improvement_px: float = 0.01,
) -> dict:
    """Vectorize raster line art as contour boundaries or centerline strokes.

    ``contour`` preserves the project's original behavior. ``centerline``
    converts dark line-art pixels to a one-pixel skeleton and emits robot-
    friendly open/closed pen trajectories.
    """

    normalized_mode = str(
        mode
    ).strip().lower()

    if normalized_mode not in _VECTOR_MODES:
        supported = ", ".join(
            sorted(
                _VECTOR_MODES
            )
        )
        raise ValueError(
            f"不支持的 vector mode：{mode}。"
            f"当前支持：{supported}"
        )

    if float(stroke_width) <= 0:
        raise ValueError(
            "stroke_width 必须为正数"
        )

    if normalized_mode == "contour":
        strokes, width, height = (
            _contour_strokes(
                line_img,
                min_area=min_area,
                epsilon_ratio=(
                    epsilon_ratio
                ),
            )
        )

        # Apply the same robot path ordering policy used by centerline
        # so pen-up comparisons are not biased by findContours order.
        strokes = order_strokes_greedy(
            strokes
        )

        output_svg = _write_svg(
            output_svg,
            width=width,
            height=height,
            strokes=strokes,
            stroke_width=stroke_width,
        )

        return {
            "mode": "contour",
            "width": width,
            "height": height,
            "path_count": len(
                strokes
            ),
            "bytes": output_svg.stat().st_size,
            **_motion_metadata(
                strokes
            ),
        }

    result = extract_centerline_strokes(
        line_img,
        CenterlineConfig(
            threshold=(
                centerline_threshold
            ),
            spur_length_px=(
                centerline_spur_length_px
            ),
            min_stroke_length_px=(
                centerline_min_stroke_length_px
            ),
            simplify_epsilon_px=(
                centerline_simplify_epsilon_px
            ),
            max_thinning_iterations=(
                centerline_max_thinning_iterations
            ),
            junction_cluster_radius_px=(
                centerline_junction_cluster_radius_px
            ),
            merge_angle_deg=(
                centerline_merge_angle_deg
            ),
            merge_lookahead_px=(
                centerline_merge_lookahead_px
            ),
            ordering_2opt_max_passes=(
                centerline_ordering_2opt_max_passes
            ),
            ordering_2opt_min_improvement_px=(
                centerline_ordering_2opt_min_improvement_px
            ),
        ),
    )

    height, width = (
        result.skeleton.shape
    )

    output_svg = _write_svg(
        output_svg,
        width=width,
        height=height,
        strokes=result.strokes,
        stroke_width=stroke_width,
    )

    return {
        "width": width,
        "height": height,
        "path_count": len(
            result.strokes
        ),
        "bytes": output_svg.stat().st_size,
        **result.to_metadata(),
    }


def _opencv_svg_to_raster(
    svg_path: Path,
    output_png: Path,
    output_width: int | None,
) -> dict:
    """Rasterize the M/L/Z-only SVG subset emitted by raster_to_svg."""

    root = ET.parse(
        svg_path
    ).getroot()

    viewbox = (
        root.attrib
        .get("viewBox", "")
        .replace(",", " ")
        .split()
    )

    if len(viewbox) == 4:
        _, _, source_width, source_height = map(
            float,
            viewbox,
        )
    else:
        source_width = float(
            root.attrib["width"]
        )
        source_height = float(
            root.attrib["height"]
        )

    target_width = int(
        output_width
        or round(source_width)
    )

    if (
        target_width <= 0
        or source_width <= 0
        or source_height <= 0
    ):
        raise ValueError(
            "SVG 尺寸无效"
        )

    target_height = max(
        1,
        round(
            source_height
            * target_width
            / source_width
        ),
    )

    scale_x = (
        target_width
        / source_width
    )
    scale_y = (
        target_height
        / source_height
    )

    canvas = np.full(
        (
            target_height,
            target_width,
            3,
        ),
        255,
        dtype=np.uint8,
    )

    path_count = 0
    open_path_count = 0
    closed_path_count = 0

    for element in root.iter():
        if (
            element.tag.rsplit(
                "}",
                1,
            )[-1]
            != "path"
        ):
            continue

        data = element.attrib.get(
            "d",
            "",
        )

        commands = re.sub(
            r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
            "",
            data,
        )

        if re.sub(
            r"[\s,MLZmlz]",
            "",
            commands,
        ):
            raise ValueError(
                "OpenCV fallback 仅支持本项目生成的 "
                "M/L/Z SVG 路径"
            )

        numbers = [
            float(value)
            for value in re.findall(
                r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
                data,
            )
        ]

        if (
            len(numbers) < 4
            or len(numbers) % 2
        ):
            continue

        points = np.asarray(
            [
                [
                    round(
                        numbers[index]
                        * scale_x
                    ),
                    round(
                        numbers[index + 1]
                        * scale_y
                    ),
                ]
                for index in range(
                    0,
                    len(numbers),
                    2,
                )
            ],
            dtype=np.int32,
        ).reshape(
            (-1, 1, 2)
        )

        stroke_width = float(
            element.attrib.get(
                "stroke-width",
                "1",
            )
        )

        thickness = max(
            1,
            round(
                stroke_width
                * (
                    scale_x
                    + scale_y
                )
                / 2
            ),
        )

        is_closed = bool(
            re.search(
                r"[Zz]",
                data,
            )
        )

        cv2.polylines(
            canvas,
            [points],
            is_closed,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

        path_count += 1
        if is_closed:
            closed_path_count += 1
        else:
            open_path_count += 1

    # A blank SVG is still a valid vectorization result. This matters for
    # centerline mode when aggressive filtering removes every tiny stroke.
    write_image(
        output_png,
        canvas,
    )

    return {
        "png": str(output_png),
        "bytes": output_png.stat().st_size,
        "backend": "opencv-fallback",
        "paths": path_count,
        "open_paths": open_path_count,
        "closed_paths": closed_path_count,
    }


def svg_to_raster(
    svg_path: str | Path,
    output_png: str | Path,
    output_width: int | None = None,
) -> dict:
    """
    优先使用 CairoSVG；
    缺少原生 Cairo 时回退到项目 SVG 专用 OpenCV 后端。
    """

    svg_path = Path(
        svg_path
    )
    output_png = Path(
        output_png
    )

    if not svg_path.exists():
        raise FileNotFoundError(
            f"找不到 SVG 文件：{svg_path}"
        )

    output_png.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cairo_error = None

    try:
        import cairosvg

        kwargs = (
            {
                "output_width":
                    int(output_width)
            }
            if output_width
            else {}
        )

        # Do not hand a Unicode Windows filename directly to CairoSVG.
        png_bytes = cairosvg.svg2png(
            bytestring=svg_path.read_bytes(),
            **kwargs,
        )

        output_png.write_bytes(
            png_bytes
        )

        return {
            "png": str(output_png),
            "bytes": output_png.stat().st_size,
            "backend": "cairosvg",
        }

    except Exception as exc:
        cairo_error = exc

    try:
        info = _opencv_svg_to_raster(
            svg_path,
            output_png,
            output_width,
        )

        info["fallback_reason"] = (
            f"{type(cairo_error).__name__}: "
            f"{cairo_error}"
        )

        return info

    except Exception as fallback_exc:
        raise RuntimeError(
            "SVG 回栅格化失败。"
            "已优先尝试 CairoSVG，并尝试本项目 SVG 专用 "
            "OpenCV fallback。"
            f"当前解释器：{sys.executable}。"
            f"CairoSVG 异常："
            f"{type(cairo_error).__name__}: "
            f"{cairo_error}；"
            f"fallback 异常："
            f"{type(fallback_exc).__name__}: "
            f"{fallback_exc}"
        ) from fallback_exc
