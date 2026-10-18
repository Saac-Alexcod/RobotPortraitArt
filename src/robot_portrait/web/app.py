from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from pydantic import BaseModel, Field

from .jobs import (
    InvalidImageError,
    JobStore,
)
from .runtime import (
    PROJECT_ROOT,
    default_generator_statuses,
    default_supported_methods,
)


GeneratorStatusProvider = Callable[
    [],
    dict[str, dict[str, Any]],
]

SupportedMethodsProvider = Callable[
    [],
    tuple[str, ...],
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


class JobCreatedResponse(BaseModel):
    job_id: str
    status: str
    method: str
    vector_mode: str
    size: int


def create_app(
    *,
    generator_status_provider: GeneratorStatusProvider | None = None,
    supported_methods_provider: SupportedMethodsProvider | None = None,
    job_store: JobStore | None = None,
    max_upload_bytes: int = 10 * 1024 * 1024,
) -> FastAPI:
    """Create the RobotPortraitArt HTTP API application."""

    if max_upload_bytes <= 0:
        raise ValueError(
            "max_upload_bytes 必须为正整数"
        )

    status_provider = (
        generator_status_provider
        or default_generator_statuses
    )
    methods_provider = (
        supported_methods_provider
        or default_supported_methods
    )
    jobs = (
        job_store
        or JobStore(
            PROJECT_ROOT
            / "outputs"
            / "web_jobs"
        )
    )

    api = FastAPI(
        title="RobotPortraitArt API",
        version="0.3.0",
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

    @api.post(
        "/api/jobs",
        response_model=JobCreatedResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["generation"],
        summary="Create a portrait generation job",
    )
    async def create_job(
        image: UploadFile = File(...),
        method: str = Form("baseline"),
        vector_mode: str = Form("contour"),
        size: int = Form(512),
    ) -> JobCreatedResponse:
        method = method.strip()
        vector_mode = vector_mode.strip()

        supported_methods = set(
            methods_provider()
        )
        if method not in supported_methods:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={
                    "code": "unsupported_method",
                    "message": (
                        f"不支持的生成方法：{method}"
                    ),
                    "supported": sorted(
                        supported_methods
                    ),
                },
            )

        if vector_mode not in {
            "contour",
            "centerline",
        }:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={
                    "code": "unsupported_vector_mode",
                    "message": (
                        "vector_mode 必须是 "
                        "contour 或 centerline"
                    ),
                },
            )

        if not 64 <= int(size) <= 2048:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={
                    "code": "invalid_size",
                    "message": (
                        "size 必须位于 64..2048"
                    ),
                },
            )

        if (
            method == "apdrawinggan"
            and int(size) != 512
        ):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={
                    "code": "apdrawinggan_size",
                    "message": (
                        "APDrawingGAN 当前要求 size=512"
                    ),
                },
            )

        payload = await image.read(
            max_upload_bytes + 1
        )

        if len(payload) > max_upload_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail={
                    "code": "upload_too_large",
                    "message": (
                        "上传图片超过大小限制"
                    ),
                    "max_upload_bytes":
                        max_upload_bytes,
                },
            )

        try:
            record = jobs.create(
                image_bytes=payload,
                original_filename=(
                    image.filename
                ),
                method=method,
                vector_mode=vector_mode,
                size=int(size),
            )
        except InvalidImageError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": "invalid_image",
                    "message": str(exc),
                },
            ) from exc

        return JobCreatedResponse(
            job_id=record.job_id,
            status=record.status,
            method=record.method,
            vector_mode=(
                record.vector_mode
            ),
            size=record.size,
        )

    return api


app = create_app()
