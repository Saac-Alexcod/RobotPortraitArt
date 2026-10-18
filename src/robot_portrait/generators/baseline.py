from __future__ import annotations

from typing import Any, Mapping

from .base import (
    GeneratorOutput,
    GeneratorRequest,
    GeneratorStatus,
    PortraitGenerator,
)
from ..lineart import portrait_lineart


class BaselineGenerator(PortraitGenerator):
    """Deterministic OpenCV line-art generator."""

    name = "baseline"

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
    ):
        self.config = dict(config or {})

    def status(self) -> GeneratorStatus:
        return GeneratorStatus(
            available=True,
            details={
                "backend": "opencv",
                "algorithm": "adaptive-threshold-lineart",
            },
        )

    def generate(
        self,
        request: GeneratorRequest,
    ) -> GeneratorOutput:
        image = portrait_lineart(
            request.prepared_image,
            **self.config,
        )

        return GeneratorOutput(
            image=image,
            metadata={
                "backend": "opencv",
                "algorithm": "adaptive-threshold-lineart",
                "parameters": dict(self.config),
                "seed": int(request.seed),
            },
        )