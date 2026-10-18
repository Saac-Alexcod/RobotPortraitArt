from __future__ import annotations

import os
from pathlib import Path

from ..generators import (
    APDrawingGANAutoPreprocessConfig,
    APDrawingGANAutoPreprocessGenerator,
    APDrawingGANGenerator,
    APDrawingGANGeneratorConfig,
    BaselineGenerator,
)
from ..pipeline import (
    GenerationPipeline,
    PipelineConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def build_pipeline(
    *,
    vector_mode: str = "contour",
    size: int = 512,
    seed: int = 42,
) -> GenerationPipeline:
    """Build one pipeline for a web job without running preflight."""

    if vector_mode not in {
        "contour",
        "centerline",
    }:
        raise ValueError(
            "vector_mode 必须是 contour 或 centerline"
        )

    base_config = PipelineConfig.from_yaml(
        PROJECT_ROOT
        / "configs"
        / "default.yaml",
        preprocess_size=size,
        seed=seed,
    )

    vector_config = dict(
        base_config.vector
    )
    vector_config[
        "mode"
    ] = vector_mode

    config = PipelineConfig(
        preprocess_size=int(size),
        seed=int(seed),
        baseline=dict(
            base_config.baseline
        ),
        vector=vector_config,
        source_path=(
            base_config.source_path
        ),
    )

    apdrawinggan_backend = (
        APDrawingGANGenerator(
            APDrawingGANGeneratorConfig(
                repo=(
                    PROJECT_ROOT
                    / "third_party"
                    / "APDrawingGAN"
                ),
                distro=os.environ.get(
                    "ROBOT_PORTRAIT_APD_DISTRO",
                    "Ubuntu-22.04",
                ),
                python_command=os.environ.get(
                    "ROBOT_PORTRAIT_APD_PYTHON",
                    "python3",
                ),
            )
        )
    )

    apdrawinggan = (
        APDrawingGANAutoPreprocessGenerator(
            apdrawinggan_backend,
            APDrawingGANAutoPreprocessConfig(
                yunet_model=(
                    PROJECT_ROOT
                    / "third_party"
                    / "models"
                    / (
                        "face_detection_"
                        "yunet_2023mar.onnx"
                    )
                ),
                humanseg_model=(
                    PROJECT_ROOT
                    / "third_party"
                    / "models"
                    / (
                        "human_segmentation_"
                        "pphumanseg_2023mar.onnx"
                    )
                ),
            ),
        )
    )

    return GenerationPipeline(
        config,
        generators={
            BaselineGenerator.name:
                BaselineGenerator(
                    config.baseline
                ),
            (
                APDrawingGANAutoPreprocessGenerator
                .name
            ):
                apdrawinggan,
        },
    )


def build_default_pipeline() -> GenerationPipeline:
    """Backward-compatible default web pipeline."""

    return build_pipeline()


def default_generator_statuses() -> dict[str, dict[str, object]]:
    """Return JSON-ready status metadata for the default web generators."""

    return (
        build_default_pipeline()
        .generator_statuses()
    )


def default_supported_methods() -> tuple[str, ...]:
    """Return registered method names without running generator preflight."""

    return (
        build_default_pipeline()
        .supported_methods
    )
