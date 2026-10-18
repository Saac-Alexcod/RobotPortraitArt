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


from robot_portrait.generators import (
    BaselineGenerator,
    GeneratorOutput,
    GeneratorRequest,
    GeneratorStatus,
    GeneratorUnavailableError,
    PortraitGenerator,
)
from robot_portrait.io import (
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


class FakeGenerator(
    PortraitGenerator
):
    name = "fake"

    def status(
        self,
    ) -> GeneratorStatus:
        return GeneratorStatus(
            available=True,
            details={
                "backend":
                    "test-double",
            },
        )

    def generate(
        self,
        request: GeneratorRequest,
    ) -> GeneratorOutput:
        height, width = (
            request
            .prepared_image
            .shape[:2]
        )

        image = np.full(
            (
                height,
                width,
            ),
            255,
            dtype=np.uint8,
        )

        y1 = max(
            1,
            height // 4,
        )

        y2 = min(
            height - 1,
            y1
            + max(
                3,
                height // 6,
            ),
        )

        x1 = max(
            1,
            width // 4,
        )

        x2 = min(
            width - 1,
            x1
            + max(
                3,
                width // 6,
            ),
        )

        image[
            y1:y2,
            x1:x2,
        ] = 0

        return GeneratorOutput(
            image=image,

            metadata={
                "backend":
                    "test-double",

                "received_seed":
                    request.seed,
            },

            warnings=(
                "synthetic generator warning",
            ),
        )


class UnavailableGenerator(
    PortraitGenerator
):
    name = "offline"

    def status(
        self,
    ) -> GeneratorStatus:
        return GeneratorStatus(
            available=False,

            reason=(
                "model checkpoint missing"
            ),

            details={
                "backend":
                    "external",
            },
        )

    def generate(
        self,
        request: GeneratorRequest,
    ) -> GeneratorOutput:
        raise AssertionError(
            "unavailable generator "
            "must never be executed"
        )


class MismatchedGenerator(
    FakeGenerator
):
    name = "different"


def test_generator_output_rejects_empty_image():
    with pytest.raises(
        ValueError,
        match="非空 numpy 数组",
    ):
        GeneratorOutput(
            image=np.asarray(
                [],
                dtype=np.uint8,
            )
        )


def test_default_pipeline_registers_baseline(
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

    assert (
        pipeline.supported_methods
        == (
            "baseline",
        )
    )

    status = (
        pipeline
        .generator_statuses()
        ["baseline"]
    )

    assert (
        status[
            "available"
        ]
        is True
    )

    assert (
        status[
            "details"
        ][
            "backend"
        ]
        == "opencv"
    )


def test_baseline_generator_contract(
    tmp_path,
):
    generator = BaselineGenerator(
        {
            "max_side": 1024,
            "bilateral_d": 5,

            "bilateral_sigma_color":
                50,

            "bilateral_sigma_space":
                50,

            "adaptive_block_size":
                11,

            "adaptive_c": 2,
            "morph_kernel": 0,
        }
    )

    image = np.full(
        (
            64,
            64,
            3,
        ),
        230,
        dtype=np.uint8,
    )

    image[
        20:44,
        20:44,
    ] = 30

    output = (
        generator.generate(
            GeneratorRequest(
                prepared_image=image,

                input_path=(
                    tmp_path
                    / "input.png"
                ),

                prepared_path=(
                    tmp_path
                    / "prepared.png"
                ),

                output_dir=(
                    tmp_path
                ),

                seed=123,
            )
        )
    )

    assert (
        output.image.shape
        == (
            64,
            64,
        )
    )

    assert (
        output.metadata[
            "backend"
        ]
        == "opencv"
    )

    assert (
        output.metadata[
            "seed"
        ]
        == 123
    )


def test_pipeline_uses_injected_generator_and_records_manifest(
    tmp_path,
):
    input_path = (
        tmp_path
        / "输入.png"
    )

    output_dir = (
        tmp_path
        / "输出"
    )

    _synthetic_portrait(
        input_path
    )

    config = (
        PipelineConfig
        .from_yaml(
            _config_file(
                tmp_path
            ),
            preprocess_size=64,
            seed=19,
        )
    )

    pipeline = GenerationPipeline(
        config,
        generators={
            "fake":
                FakeGenerator(),
        },
    )

    result = pipeline.run(
        input_path,
        output_dir,
        method="fake",
    )

    manifest = json.loads(
        result.manifest_path
        .read_text(
            encoding="utf-8"
        )
    )

    assert (
        result.method
        == "fake"
    )

    assert (
        result.raster_path.name
        == "fake.png"
    )

    assert (
        manifest[
            "generation"
        ][
            "name"
        ]
        == "fake"
    )

    assert (
        manifest[
            "generation"
        ][
            "status"
        ][
            "available"
        ]
        is True
    )

    assert (
        manifest[
            "generation"
        ][
            "metadata"
        ][
            "backend"
        ]
        == "test-double"
    )

    assert (
        manifest[
            "generation"
        ][
            "metadata"
        ][
            "received_seed"
        ]
        == 19
    )

    assert (
        "synthetic generator warning"
        in manifest[
            "warnings"
        ]
    )


def test_unavailable_generator_clears_stale_managed_artifacts(
    tmp_path,
):
    config = (
        PipelineConfig
        .from_yaml(
            _config_file(
                tmp_path
            ),
            preprocess_size=64,
        )
    )

    pipeline = GenerationPipeline(
        config,
        generators={
            "offline":
                UnavailableGenerator(),
        },
    )

    output_dir = (
        tmp_path
        / "out"
    )

    output_dir.mkdir()

    (
        output_dir
        / "manifest.json"
    ).write_text(
        "stale",
        encoding="utf-8",
    )

    (
        output_dir
        / "offline.png"
    ).write_bytes(
        b"stale"
    )

    with pytest.raises(
        GeneratorUnavailableError,
        match="checkpoint missing",
    ):
        pipeline.run(
            tmp_path
            / "missing.png",

            output_dir,

            method="offline",
        )

    assert output_dir.exists()

    assert not (
        output_dir
        / "manifest.json"
    ).exists()

    assert not (
        output_dir
        / "offline.png"
    ).exists()


def test_registry_rejects_mismatched_generator_name(
    tmp_path,
):
    config = (
        PipelineConfig
        .from_yaml(
            _config_file(
                tmp_path
            )
        )
    )

    with pytest.raises(
        ValueError,
        match="注册名",
    ):
        GenerationPipeline(
            config,

            generators={
                "fake":
                    MismatchedGenerator(),
            },
        )


def test_switching_methods_cleans_old_managed_artifacts(
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

    config = (
        PipelineConfig
        .from_yaml(
            _config_file(
                tmp_path
            ),
            preprocess_size=64,
        )
    )

    pipeline = GenerationPipeline(
        config,

        generators={
            "baseline":
                BaselineGenerator(
                    config.baseline
                ),

            "fake":
                FakeGenerator(),
        },
    )

    pipeline.run(
        input_path,
        output_dir,
        method="baseline",
    )

    assert (
        output_dir
        / "baseline.png"
    ).exists()

    assert (
        output_dir
        / "baseline.svg"
    ).exists()

    pipeline.run(
        input_path,
        output_dir,
        method="fake",
    )

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
        / (
            "baseline"
            "_vector_render.png"
        )
    ).exists()

    assert (
        output_dir
        / "fake.png"
    ).exists()

    assert (
        output_dir
        / "fake.svg"
    ).exists()