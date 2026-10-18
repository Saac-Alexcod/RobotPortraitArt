from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src"),
)


from robot_portrait.generators import (
    GeneratorOutput,
    GeneratorRequest,
    GeneratorStatus,
    PortraitGenerator,
    PreprocessTransform,
)
from robot_portrait.io import (
    write_image,
)
from robot_portrait.pipeline import (
    GenerationPipeline,
    PipelineConfig,
)


class RecordingGenerator(
    PortraitGenerator
):
    name = "recording"

    def __init__(
        self,
    ) -> None:
        self.last_request: (
            GeneratorRequest
            | None
        ) = None

    def status(
        self,
    ) -> GeneratorStatus:
        return GeneratorStatus(
            available=True,
        )

    def generate(
        self,
        request: GeneratorRequest,
    ) -> GeneratorOutput:
        self.last_request = request

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

        image[
            height // 4:
            3 * height // 4,
            width // 2:
            width // 2 + 2,
        ] = 0

        return GeneratorOutput(
            image=image,
            metadata={
                "backend":
                    "recording",
            },
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


def _input_image(
    tmp_path: Path,
) -> Path:
    path = (
        tmp_path
        / "原始 人像.png"
    )

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
    ] = 210

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

    return path


def _pipeline(
    tmp_path: Path,
    generator: RecordingGenerator,
) -> GenerationPipeline:
    config = (
        PipelineConfig
        .from_yaml(
            _config_file(
                tmp_path
            ),
            preprocess_size=64,
            seed=23,
        )
    )

    return GenerationPipeline(
        config,
        generators={
            generator.name:
                generator,
        },
    )


def test_pipeline_forwards_preprocess_transform_and_auxiliary_inputs(
    tmp_path,
):
    input_path = (
        _input_image(
            tmp_path
        )
    )

    landmark = (
        tmp_path
        / "五点.txt"
    )

    mask = (
        tmp_path
        / "背景.png"
    )

    landmark.write_text(
        "placeholder",
        encoding="utf-8",
    )

    mask.write_bytes(
        b"placeholder"
    )

    generator = (
        RecordingGenerator()
    )

    result = _pipeline(
        tmp_path,
        generator,
    ).run(
        input_path,
        tmp_path / "输出",
        method="recording",
        auxiliary_inputs={
            "landmark":
                str(
                    landmark
                ),
            "mask":
                mask,
        },
    )

    request = (
        generator.last_request
    )

    assert request is not None

    assert isinstance(
        request.preprocess_transform,
        PreprocessTransform,
    )

    transform = (
        request
        .preprocess_transform
    )

    assert (
        transform.original_size
        == (
            72,
            96,
        )
    )

    assert (
        transform.output_size
        == (
            64,
            64,
        )
    )

    crop = (
        result.preprocessing[
            "crop_box"
        ]
    )

    assert (
        transform.crop_box
        == (
            crop["left"],
            crop["top"],
            crop["right"],
            crop["bottom"],
        )
    )

    assert (
        request.auxiliary_inputs
        == {
            "landmark":
                landmark,
            "mask":
                mask,
        }
    )


def test_pipeline_defaults_to_empty_auxiliary_inputs(
    tmp_path,
):
    generator = (
        RecordingGenerator()
    )

    _pipeline(
        tmp_path,
        generator,
    ).run(
        _input_image(
            tmp_path
        ),
        tmp_path / "输出",
        method="recording",
    )

    request = (
        generator.last_request
    )

    assert request is not None

    assert (
        request.auxiliary_inputs
        == {}
    )

    assert isinstance(
        request.preprocess_transform,
        PreprocessTransform,
    )
