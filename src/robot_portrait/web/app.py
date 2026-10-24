from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
import json
import time
from threading import Lock
from pathlib import Path
from typing import Any

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .jobs import (
    InvalidImageError,
    JobNotFoundError,
    JobStore,
)
from .runtime import (
    PROJECT_ROOT,
    build_pipeline,
    default_generator_statuses,
    default_supported_methods,
)
from .worker import (
    JobWorker,
    JobWorkerLike,
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


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    method: str
    vector_mode: str
    size: int
    original_filename: str | None
    input_file: str
    created_at: str
    updated_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    error: dict[str, str] | None = None
    result_files: dict[str, str] = Field(
        default_factory=dict
    )


def _job_not_found(
    job_id: str,
) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={
            "code": "job_not_found",
            "message": f"找不到 job：{job_id}",
        },
    )




def _effective_method_metadata(
    name: str,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return method status as it applies to the Web workflow.

    APDrawingGAN's reusable wrapper can remain available for manual
    landmark/mask input even when the automatic YuNet/PPHumanSeg models are
    missing.  The Web UI does not expose those manual auxiliary inputs, so its
    method availability must also require automatic preprocessing.
    """

    effective = dict(metadata or {})
    effective.setdefault(
        "available",
        False,
    )
    effective.setdefault(
        "reason",
        None,
    )
    raw_details = effective.get(
        "details"
    )
    effective[
        "details"
    ] = (
        dict(raw_details)
        if isinstance(
            raw_details,
            dict,
        )
        else {}
    )

    if name != "apdrawinggan":
        return effective

    auto = effective[
        "details"
    ].get(
        "auto_preprocess"
    )

    if (
        bool(effective.get("available"))
        and not (
            isinstance(auto, dict)
            and auto.get("available") is True
        )
    ):
        effective[
            "available"
        ] = False
        effective[
            "reason"
        ] = (
            "APDrawingGAN 自动预处理不可用："
            "请检查 YuNet 与 PPHumanSeg 模型。"
        )

    return effective


def _media_type_for(
    path: Path,
) -> str:
    suffix = path.suffix.lower()

    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".svg": "image/svg+xml",
        ".json": "application/json",
    }.get(
        suffix,
        "application/octet-stream",
    )


def create_app(
    *,
    generator_status_provider: GeneratorStatusProvider | None = None,
    supported_methods_provider: SupportedMethodsProvider | None = None,
    job_store: JobStore | None = None,
    job_worker: JobWorkerLike | None = None,
    max_upload_bytes: int = 10 * 1024 * 1024,
) -> FastAPI:
    """Create the RobotPortraitArt HTTP API application."""

    if max_upload_bytes <= 0:
        raise ValueError(
            "max_upload_bytes 必须为正整数"
        )

    # WSL/CUDA preflight can take seconds. Share its result between the
    # methods panel and job submission, but expire it promptly. The worker's
    # execution-time check remains authoritative.
    _status_lock = Lock()
    _cached_statuses: dict[str, dict[str, Any]] | None = None
    _cache_until = 0.0

    def _default_statuses_cached(*, refresh: bool = False) -> dict[str, dict[str, Any]]:
        nonlocal _cached_statuses, _cache_until
        with _status_lock:
            now = time.monotonic()
            if not refresh and _cached_statuses is not None and now < _cache_until:
                return _cached_statuses
            result = default_generator_statuses()
            # Shorter negative cache so newly installed models recover fast.
            apd = result.get("apdrawinggan", {})
            ttl = 30.0 if _effective_method_metadata("apdrawinggan", apd).get("available") else 5.0
            _cached_statuses = result
            _cache_until = time.monotonic() + ttl
            return result

    status_provider = generator_status_provider or _default_statuses_cached
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
    worker = (
        job_worker
        or JobWorker(
            jobs,
            pipeline_builder=(
                build_pipeline
            ),
        )
    )

    @asynccontextmanager
    async def lifespan(
        _: FastAPI,
    ):
        # Only recover existing persisted jobs at app/process startup; do not
        # change the standalone JobWorker.start() semantics, as clients may
        # enqueue jobs immediately before explicitly starting a worker.
        if isinstance(worker, JobWorker):
            jobs.fail_interrupted_jobs()
        worker.start()
        try:
            yield
        finally:
            worker.stop(
                wait=True
            )

    api = FastAPI(
        title="RobotPortraitArt API",
        version="0.4.0",
        description=(
            "HTTP API for portrait generation, vectorization, "
            "and robot drawing path planning."
        ),
        lifespan=lifespan,
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
    def methods(refresh: bool = False) -> MethodsResponse:
        if refresh and generator_status_provider is None:
            statuses = _default_statuses_cached(refresh=True)
        else:
            statuses = status_provider()

        return MethodsResponse(
            methods=[
                MethodStatus(
                    name=name,
                    **_effective_method_metadata(
                        name,
                        metadata,
                    ),
                )
                for name, metadata
                in sorted(
                    statuses.items()
                )
            ]
        )

    @api.get(
        "/api/jobs/{job_id}",
        response_model=JobStatusResponse,
        tags=["generation"],
        summary="Get generation job status",
    )
    def get_job(
        job_id: str,
    ) -> JobStatusResponse:
        try:
            record = jobs.get(
                job_id
            )
        except JobNotFoundError as exc:
            raise _job_not_found(
                job_id
            ) from exc

        return JobStatusResponse(
            **record.to_metadata()
        )

    @api.get(
        "/api/jobs/{job_id}/manifest",
        tags=["generation"],
        summary="Get completed job manifest",
    )
    def get_job_manifest(
        job_id: str,
    ) -> dict[str, Any]:
        try:
            record = jobs.get(
                job_id
            )
        except JobNotFoundError as exc:
            raise _job_not_found(
                job_id
            ) from exc

        if record.status != "completed":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "manifest_unavailable",
                    "message": (
                        "job 尚未完成，manifest 当前不可用"
                    ),
                    "status": record.status,
                },
            )

        try:
            path = jobs.manifest_path(
                job_id
            )
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "manifest_not_found",
                    "message": str(exc),
                },
            ) from exc

        try:
            payload = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
        except (
            OSError,
            json.JSONDecodeError,
        ) as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "code": "manifest_invalid",
                    "message": (
                        "manifest 无法读取或不是合法 JSON"
                    ),
                },
            ) from exc

        if not isinstance(
            payload,
            dict,
        ):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "code": "manifest_invalid",
                    "message": (
                        "manifest 根节点必须是 JSON object"
                    ),
                },
            )

        return payload

    @api.get(
        "/api/jobs/{job_id}/files/{filename}",
        response_class=FileResponse,
        tags=["generation"],
        summary="Get a public job file",
    )
    def get_job_file(
        job_id: str,
        filename: str,
    ) -> FileResponse:
        try:
            path = jobs.public_file_path(
                job_id,
                filename,
            )
        except JobNotFoundError as exc:
            raise _job_not_found(
                job_id
            ) from exc
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "file_not_available",
                    "message": str(exc),
                },
            ) from exc

        return FileResponse(
            path=path,
            media_type=_media_type_for(
                path
            ),
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

        # Baseline is an in-process OpenCV generator and has no external
        # preflight.  Only APDrawingGAN needs an availability gate here; this
        # keeps the Baseline fallback responsive even when WSL/CUDA is down.
        if method == "apdrawinggan":
            method_status = (
                _effective_method_metadata(
                    method,
                    (await run_in_threadpool(status_provider)).get(
                        method
                    ),
                )
            )

            if not method_status[
                "available"
            ]:
                raise HTTPException(
                    status_code=(
                        status.HTTP_503_SERVICE_UNAVAILABLE
                    ),
                    detail={
                        "code": "method_unavailable",
                        "message": (
                            method_status.get(
                                "reason"
                            )
                            or (
                                "APDrawingGAN 当前不可用"
                            )
                        ),
                        "method": method,
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

        try:
            worker.submit(
                record.job_id
            )
        except RuntimeError as exc:
            jobs.mark_failed(
                record.job_id,
                exc,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": "worker_unavailable",
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
