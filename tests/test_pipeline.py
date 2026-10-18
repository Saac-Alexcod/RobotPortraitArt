from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src"),
)


from robot_portrait.io import (
    read_image,
    write_image,
)

from robot_portrait.pipeline import (
    GenerationPipeline,
    PipelineConfig,
)


def _config_file(
    tmp_path: Path,
) -> Path:
    path = (
        tmp_path
        / "config.yaml"
    )

    path.write_text(
        """
baseline:
  max_side: 1024
  bilateral_d: 5
  bilateral_sigma_color: 50
  bilateral_sigma_space: 50
  adaptive_block_size: 11
  adaptive_c: 2
  morph_kernel: 0

vector:
  min_area: 1.0
  epsilon_ratio: 0.0025
  stroke_width: 1.2
""".strip(),
        encoding="utf-8",
    )

    return path


def _synthetic_portrait(
    path: Path,
) -> None:
    image = np.full(
        (
            96,
            72,
            3,
        ),
        245,
        dtype=np.uint8,
    )

    image[
        20:76,
        18:54,
    ] = (
        210,
        210,
        210,
    )

    image[
        36:40,
        26:30,
    ] = 20

    image[
        36:40,
        42:46,
    ] = 20

    image[
        56:60,
        30:43,
    ] = 30

    write_image(
        path,
        image,
    )


def test_pipeline_config_rejects_missing_sections(
    tmp_path,
):
    path = (
        tmp_path
        / "invalid.yaml"
    )

    path.write_text(
        "baseline: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            "baseline 和 vector"
        ),
    ):
        PipelineConfig.from_yaml(
            path
        )


def test_pipeline_config_rejects_non_positive_size(
    tmp_path,
):
    with pytest.raises(
        ValueError,
        match=(
            "preprocess_size "
            "必须是正整数"
        ),
    ):
        PipelineConfig.from_yaml(
            _config_file(
                tmp_path
            ),
            preprocess_size=0,
        )


def test_pipeline_rejects_unknown_method(
    tmp_path,
):
    pipeline = (
        GenerationPipeline
        .from_yaml(
            _config_file(
                tmp_path
            )
        )
    )

    with pytest.raises(
        ValueError,
        match="不支持的生成方法",
    ):
        pipeline.run(
            tmp_path
            / "missing.png",

            tmp_path
            / "out",

            method="gan",
        )


def test_pipeline_end_to_end_unicode_paths(
    tmp_path,
):
    input_path = (
        tmp_path
        / "输入 人像.png"
    )

    output_dir = (
        tmp_path
        / "实验 输出"
    )

    _synthetic_portrait(
        input_path
    )

    pipeline = (
        GenerationPipeline
        .from_yaml(
            _config_file(
                tmp_path
            ),
            preprocess_size=64,
            seed=7,
        )
    )

    result = pipeline.run(
        input_path,
        output_dir,
    )

    assert (
        result
        .prepared_path
        .exists()
    )

    assert (
        result
        .raster_path
        .exists()
    )

    assert (
        result
        .svg_path
        .exists()
    )

    assert (
        result
        .vector_render_path
        .exists()
    )

    assert (
        result
        .manifest_path
        .exists()
    )

    assert (
        read_image(
            result.prepared_path
        ).shape
        == (
            64,
            64,
            3,
        )
    )

    assert (
        read_image(
            result.raster_path
        ).shape[:2]
        == (
            64,
            64,
        )
    )

    assert (
        read_image(
            result
            .vector_render_path
        ).shape
        == (
            64,
            64,
            3,
        )
    )

    manifest = json.loads(
        result
        .manifest_path
        .read_text(
            encoding="utf-8"
        )
    )

    assert (
        manifest["status"]
        == "completed"
    )

    assert (
        manifest["method"]
        == "baseline"
    )

    assert (
        manifest["files"]
        ["prepared"]
        == "prepared.png"
    )

    assert (
        manifest["files"]
        ["vector_svg"]
        == "baseline.svg"
    )

    assert (
        manifest[
            "preprocessing"
        ][
            "output_size"
        ]
        == {
            "width": 64,
            "height": 64,
        }
    )

    assert (
        manifest[
            "config"
        ][
            "seed"
        ]
        == 7
    )

    assert set(
        manifest[
            "timings_ms"
        ]
    ) == {
        "preprocess",
        "generate",
        "vectorize",
        "rasterize",
        "total",
    }

    assert all(
        value >= 0
        for value
        in manifest[
            "timings_ms"
        ].values()
    )

    assert not (
        output_dir
        / "manifest.json.tmp"
    ).exists()


def test_pipeline_is_deterministic_for_same_input_and_config(
    tmp_path,
):
    input_path = (
        tmp_path
        / "portrait.png"
    )

    _synthetic_portrait(
        input_path
    )

    pipeline = (
        GenerationPipeline
        .from_yaml(
            _config_file(
                tmp_path
            ),
            preprocess_size=64,
            seed=42,
        )
    )

    first = pipeline.run(
        input_path,
        tmp_path / "first",
    )

    second = pipeline.run(
        input_path,
        tmp_path / "second",
    )

    assert (
        first
        .raster_path
        .read_bytes()
        ==
        second
        .raster_path
        .read_bytes()
    )

    assert (
        first
        .svg_path
        .read_text(
            encoding="utf-8"
        )
        ==
        second
        .svg_path
        .read_text(
            encoding="utf-8"
        )
    )

    assert (
        first
        .vector_render_path
        .read_bytes()
        ==
        second
        .vector_render_path
        .read_bytes()
    )


def test_pipeline_rejects_corrupt_input(
    tmp_path,
):
    corrupt = (
        tmp_path
        / "坏图.png"
    )

    corrupt.write_bytes(
        b"not-an-image"
    )

    pipeline = (
        GenerationPipeline
        .from_yaml(
            _config_file(
                tmp_path
            ),
            preprocess_size=64,
        )
    )

    with pytest.raises(
        ValueError,
        match="无法读取输入照片",
    ):
        pipeline.run(
            corrupt,
            tmp_path / "out",
        )


def test_failed_rerun_does_not_leave_stale_success_manifest(
    tmp_path,
):
    input_path = (
        tmp_path
        / "portrait.png"
    )

    output_dir = (
        tmp_path
        / "shared"
    )

    _synthetic_portrait(
        input_path
    )

    pipeline = (
        GenerationPipeline
        .from_yaml(
            _config_file(
                tmp_path
            ),
            preprocess_size=64,
        )
    )

    first = pipeline.run(
        input_path,
        output_dir,
    )

    assert (
        first
        .manifest_path
        .exists()
    )

    input_path.write_bytes(
        b"broken"
    )

    with pytest.raises(
        ValueError,
        match="无法读取输入照片",
    ):
        pipeline.run(
            input_path,
            output_dir,
        )

    assert not (
        output_dir
        / "manifest.json"
    ).exists()

    assert not (
        output_dir
        / "prepared.png"
    ).exists()

    assert not (
        output_dir
        / "baseline.png"
    ).exists()

    assert not (
        output_dir
        / "baseline.svg"
    ).exists()

    assert not (
        output_dir
        / "baseline_vector_render.png"
    ).exists()