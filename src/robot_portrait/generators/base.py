from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class GeneratorRequest:
    """Input contract shared by all portrait generators."""

    prepared_image: np.ndarray = field(repr=False)
    input_path: Path
    prepared_path: Path
    output_dir: Path
    seed: int

    def __post_init__(self) -> None:
        image = self.prepared_image

        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError(
                "prepared_image 必须是非空 numpy 数组"
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

    def __post_init__(self) -> None:
        image = self.image

        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError(
                "generator output image 必须是非空 numpy 数组"
            )


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