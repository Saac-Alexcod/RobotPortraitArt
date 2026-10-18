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
from ..pipeline import GenerationPipeline, PipelineConfig


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def build_default_pipeline() -> GenerationPipeline:
    """Build the default web pipeline without running generator preflight."""

    config = PipelineConfig.from_yaml(
        PROJECT_ROOT / "configs" / "default.yaml",
        preprocess_size=512,
        seed=42,
    )

    apdrawinggan_backend = APDrawingGANGenerator(
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

    apdrawinggan = APDrawingGANAutoPreprocessGenerator(
        apdrawinggan_backend,
        APDrawingGANAutoPreprocessConfig(
            yunet_model=(
                PROJECT_ROOT
                / "third_party"
                / "models"
                / "face_detection_yunet_2023mar.onnx"
            ),
            humanseg_model=(
                PROJECT_ROOT
                / "third_party"
                / "models"
                / "human_segmentation_pphumanseg_2023mar.onnx"
            ),
        ),
    )

    return GenerationPipeline(
        config,
        generators={
            BaselineGenerator.name:
                BaselineGenerator(
                    config.baseline
                ),
            APDrawingGANAutoPreprocessGenerator.name:
                apdrawinggan,
        },
    )


def default_generator_statuses() -> dict[str, dict[str, object]]:
    """Return JSON-ready status metadata for the default web generators."""

    return build_default_pipeline().generator_statuses()
