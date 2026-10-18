from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(
        0,
        str(SRC),
    )


from robot_portrait.vectorize import (
    raster_to_svg,
    svg_to_raster,
)


def _read_image(path: Path) -> np.ndarray:
    data = np.fromfile(
        path,
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        data,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            f"无法读取图像：{path}"
        )

    return image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest().upper()


def _ratio(
    numerator: float,
    denominator: float,
) -> float | None:
    if denominator == 0:
        return None

    return float(
        numerator
        / denominator
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "在同一张固定 raster 上公平比较 contour 与 centerline。"
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--render-width",
        type=int,
        default=512,
    )

    args = parser.parse_args()

    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()

    if not input_path.exists():
        raise FileNotFoundError(
            f"找不到输入图像：{input_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = _read_image(
        input_path
    )

    results: dict[
        str,
        dict[str, object],
    ] = {}

    for mode in (
        "contour",
        "centerline",
    ):
        svg_path = (
            output_dir
            / f"{mode}.svg"
        )

        png_path = (
            output_dir
            / f"{mode}_render.png"
        )

        started = time.perf_counter()

        vector_info = raster_to_svg(
            image,
            svg_path,
            mode=mode,
        )

        vectorize_ms = (
            time.perf_counter()
            - started
        ) * 1000.0

        render_info = svg_to_raster(
            svg_path,
            png_path,
            output_width=(
                args.render_width
            ),
        )

        results[
            mode
        ] = {
            "vectorization": (
                vector_info
            ),
            "vectorize_ms": float(
                vectorize_ms
            ),
            "render": (
                render_info
            ),
            "svg": str(
                svg_path
            ),
            "render_png": str(
                png_path
            ),
        }

    contour = results[
        "contour"
    ][
        "vectorization"
    ]

    centerline = results[
        "centerline"
    ][
        "vectorization"
    ]

    comparison = {
        "input": str(
            input_path
        ),
        "input_sha256": _sha256(
            input_path
        ),
        "contour": results[
            "contour"
        ],
        "centerline": results[
            "centerline"
        ],
        "ratios_centerline_over_contour": {
            "stroke_count": _ratio(
                float(
                    centerline[
                        "stroke_count"
                    ]
                ),
                float(
                    contour[
                        "stroke_count"
                    ]
                ),
            ),
            "pen_down_length_px": _ratio(
                float(
                    centerline[
                        "pen_down_length_px"
                    ]
                ),
                float(
                    contour[
                        "pen_down_length_px"
                    ]
                ),
            ),
            "pen_up_length_px": _ratio(
                float(
                    centerline[
                        "pen_up_length_px"
                    ]
                ),
                float(
                    contour[
                        "pen_up_length_px"
                    ]
                ),
            ),
            "svg_bytes": _ratio(
                float(
                    centerline[
                        "bytes"
                    ]
                ),
                float(
                    contour[
                        "bytes"
                    ]
                ),
            ),
        },
    }

    comparison_path = (
        output_dir
        / "comparison.json"
    )

    comparison_path.write_text(
        json.dumps(
            comparison,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            comparison,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
