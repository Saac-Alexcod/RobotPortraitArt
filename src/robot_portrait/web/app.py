from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .runtime import default_generator_statuses


GeneratorStatusProvider = Callable[
    [],
    dict[str, dict[str, Any]],
]


class MethodStatus(BaseModel):
    name: str
    available: bool
    reason: str | None = None
    details: dict[str, Any] = Field(
        default_factory=dict
    )


class MethodsResponse(BaseModel):
    methods: list[MethodStatus]


def create_app(
    *,
    generator_status_provider: GeneratorStatusProvider | None = None,
) -> FastAPI:
    """Create the RobotPortraitArt HTTP API application."""

    status_provider = (
        generator_status_provider
        or default_generator_statuses
    )

    api = FastAPI(
        title="RobotPortraitArt API",
        version="0.2.0",
        description=(
            "HTTP API for portrait generation, vectorization, "
            "and robot drawing path planning."
        ),
    )

    @api.get(
        "/api/health",
        tags=["system"],
        summary="Health check",
    )
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "service": "robot-portrait-api",
        }

    @api.get(
        "/api/methods",
        response_model=MethodsResponse,
        tags=["generation"],
        summary="List portrait generation methods",
    )
    def methods() -> MethodsResponse:
        statuses = status_provider()

        return MethodsResponse(
            methods=[
                MethodStatus(
                    name=name,
                    available=bool(
                        metadata.get(
                            "available",
                            False,
                        )
                    ),
                    reason=metadata.get(
                        "reason"
                    ),
                    details=dict(
                        metadata.get(
                            "details",
                            {},
                        )
                    ),
                )
                for name, metadata
                in sorted(
                    statuses.items()
                )
            ]
        )

    return api


app = create_app()
