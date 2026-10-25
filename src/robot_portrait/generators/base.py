from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class PreprocessTransform:
    """Geometry applied by the shared preprocessing stage."""

    original_size: tuple[int, int]
    output_size: tuple[int, int]
    crop_box: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        original_width, original_height = self.original_size
        output_width, output_height = self.output_size
        left, top, right, bottom = self.crop_box

        if min(
            original_width,
            original_height,
            output_width,
            output_height,
        ) <= 0:
            raise ValueError(
                "preprocess transform 尺寸必须为正数"
            )

        if not (
            0 <= left < right <= original_width
            and 0 <= top < bottom <= original_height
        ):
            raise ValueError(
                "preprocess transform crop_box 非法"
            )


@dataclass(frozen=True, slots=True)
class GeneratorRequest:
    """Input contract shared by all portrait generators."""

    prepared_image: np.ndarray = field(repr=False)
    input_path: Path
    prepared_path: Path
    output_dir: Path
    seed: int
    preprocess_transform: PreprocessTransform | None = None
    auxiliary_inputs: Mapping[str, Path] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        image = self.prepared_image
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError(
                "prepared_image 必须是非空 numpy 数组"
            )

        if self.preprocess_transform is not None:
            expected_width, expected_height = (
                self.preprocess_transform.output_size
            )
            actual_height, actual_width = image.shape[:2]

            if (
                actual_width,
                actual_height,
            ) != (
                expected_width,
                expected_height,
            ):
                raise ValueError(
                    "prepared_image 尺寸与 preprocess_transform "
                    "output_size 不一致"
                )


@dataclass(frozen=True, slots=True)
class GeneratorStatus:
    """Availability information for CLI/API/UI preflight checks."""

    available: bool
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class GeneratorOutput:
    """Raster result and generator-specific metadata."""

    image: np.ndarray = field(repr=False)
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    vector_image: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        image = self.image
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError(
                "generator output image 必须是非空 numpy 数组"
            )
        if self.vector_image is not None and (
            not isinstance(self.vector_image, np.ndarray)
            or self.vector_image.size == 0
            or self.vector_image.shape[:2] != image.shape[:2]
        ):
            raise ValueError("vector_image 必须与肖像图像尺寸相同")


class GeneratorUnavailableError(RuntimeError):
    """Generator cannot run in the current environment."""


class PortraitGenerator(ABC):
    """Stable interface implemented by every portrait generator."""

    name: str

    @abstractmethod
    def status(self) -> GeneratorStatus:
        """Return whether this generator can currently run."""
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        request: GeneratorRequest,
    ) -> GeneratorOutput:
        """Generate one raster portrait from a preprocessed request."""
        raise NotImplementedError
