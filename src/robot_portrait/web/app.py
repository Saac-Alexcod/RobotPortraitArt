from __future__ import annotations

from fastapi import FastAPI


def create_app() -> FastAPI:
    """Create the RobotPortraitArt HTTP API application."""

    api = FastAPI(
        title="RobotPortraitArt API",
        version="0.1.0",
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

    return api


app = create_app()
