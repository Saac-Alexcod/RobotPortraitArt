from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
)
from threading import RLock
from typing import Protocol

from ..pipeline import GenerationPipeline
from .jobs import (
    JobNotFoundError,
    JobStore,
)


PipelineBuilder = Callable[
    ...,
    GenerationPipeline,
]


class JobWorkerLike(Protocol):
    def start(self) -> None:
        ...

    def submit(
        self,
        job_id: str,
    ) -> Future[None] | None:
        ...

    def stop(
        self,
        *,
        wait: bool = True,
    ) -> None:
        ...


class JobWorker:
    """Execute persisted jobs serially in one background worker thread."""

    def __init__(
        self,
        store: JobStore,
        *,
        pipeline_builder: PipelineBuilder,
    ) -> None:
        self.store = store
        self.pipeline_builder = (
            pipeline_builder
        )
        self._lock = RLock()
        self._executor: (
            ThreadPoolExecutor
            | None
        ) = None

    def start(self) -> None:
        with self._lock:
            if self._executor is None:
                self._executor = (
                    ThreadPoolExecutor(
                        max_workers=1,
                        thread_name_prefix=(
                            "robot-portrait-worker"
                        ),
                    )
                )

    def submit(
        self,
        job_id: str,
    ) -> Future[None]:
        with self._lock:
            executor = self._executor

            if executor is None:
                raise RuntimeError(
                    "job worker 尚未启动"
                )

            return executor.submit(
                self._execute_job,
                job_id,
            )

    def stop(
        self,
        *,
        wait: bool = True,
    ) -> None:
        with self._lock:
            executor = self._executor
            self._executor = None

        if executor is not None:
            executor.shutdown(
                wait=wait,
                cancel_futures=False,
            )

    def _execute_job(
        self,
        job_id: str,
    ) -> None:
        try:
            record = (
                self.store
                .mark_running(
                    job_id
                )
            )

            job_dir = (
                self.store
                .job_dir(
                    job_id
                )
            )
            input_path = (
                job_dir
                / record.input_file
            )

            pipeline = (
                self.pipeline_builder(
                    vector_mode=(
                        record.vector_mode
                    ),
                    size=record.size,
                )
            )

            result = pipeline.run(
                input_path,
                job_dir,
                method=record.method,
            )

            self.store.mark_completed(
                job_id,
                result_files={
                    "prepared":
                        result
                        .prepared_path
                        .name,
                    "raster_portrait":
                        result
                        .raster_path
                        .name,
                    "vector_svg":
                        result
                        .svg_path
                        .name,
                    "vector_render":
                        result
                        .vector_render_path
                        .name,
                    "manifest":
                        result
                        .manifest_path
                        .name,
                },
            )

        except JobNotFoundError:
            return

        except Exception as exc:
            try:
                self.store.mark_failed(
                    job_id,
                    exc,
                )
            except JobNotFoundError:
                pass
