from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from threading import RLock
import uuid

import cv2
import numpy as np

from ..io import write_image


JOB_STATUSES = {
    "queued",
    "running",
    "completed",
    "failed",
}

_JOB_ID_PATTERN = re.compile(
    r"^[0-9a-f]{32}$"
)


def _utc_now() -> str:
    return (
        datetime.now(
            timezone.utc
        )
        .isoformat()
    )


@dataclass(frozen=True, slots=True)
class JobRecord:
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
    result_files: dict[str, str] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not _JOB_ID_PATTERN.fullmatch(
            self.job_id
        ):
            raise ValueError(
                "job_id 必须是 32 位小写十六进制字符串"
            )
        if self.status not in JOB_STATUSES:
            raise ValueError(
                f"非法 job status：{self.status}"
            )
        if int(self.size) <= 0:
            raise ValueError(
                "job size 必须为正整数"
            )

    def to_metadata(self) -> dict[str, object]:
        return asdict(self)


class InvalidImageError(ValueError):
    """Uploaded bytes are not a decodable portrait image."""


class JobNotFoundError(FileNotFoundError):
    """Requested job does not exist."""


class JobStore:
    """Filesystem-backed storage for web generation jobs."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._lock = RLock()

    @staticmethod
    def _decode_image(data: bytes) -> np.ndarray:
        if not data:
            raise InvalidImageError(
                "上传文件为空"
            )

        encoded = np.frombuffer(
            data,
            dtype=np.uint8,
        )
        image = cv2.imdecode(
            encoded,
            cv2.IMREAD_COLOR,
        )

        if image is None or image.size == 0:
            raise InvalidImageError(
                "上传内容不是 OpenCV 可解码的图片"
            )

        return image

    @staticmethod
    def _write_json_atomic(
        path: Path,
        payload: dict[str, object],
    ) -> None:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        temporary = path.with_name(
            path.name + ".tmp"
        )
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary.replace(path)

    def job_dir(
        self,
        job_id: str,
    ) -> Path:
        if not _JOB_ID_PATTERN.fullmatch(
            str(job_id)
        ):
            raise JobNotFoundError(
                f"非法 job_id：{job_id}"
            )
        return self.root / str(job_id)

    def get(
        self,
        job_id: str,
    ) -> JobRecord:
        metadata_path = (
            self.job_dir(job_id)
            / "job.json"
        )

        with self._lock:
            if not metadata_path.is_file():
                raise JobNotFoundError(
                    f"找不到 job：{job_id}"
                )

            payload = json.loads(
                metadata_path.read_text(
                    encoding="utf-8"
                )
            )

        return JobRecord(
            **payload
        )

    def _replace(
        self,
        record: JobRecord,
    ) -> JobRecord:
        metadata_path = (
            self.job_dir(
                record.job_id
            )
            / "job.json"
        )

        with self._lock:
            self._write_json_atomic(
                metadata_path,
                record.to_metadata(),
            )

        return record

    def create(
        self,
        *,
        image_bytes: bytes,
        original_filename: str | None,
        method: str,
        vector_mode: str,
        size: int,
    ) -> JobRecord:
        image = self._decode_image(
            image_bytes
        )

        job_id = uuid.uuid4().hex
        job_dir = self.root / job_id

        with self._lock:
            job_dir.mkdir(
                parents=True,
                exist_ok=False,
            )

        input_path = job_dir / "input.png"

        write_image(
            input_path,
            image,
        )

        now = _utc_now()

        record = JobRecord(
            job_id=job_id,
            status="queued",
            method=method,
            vector_mode=vector_mode,
            size=int(size),
            original_filename=(
                original_filename
                or None
            ),
            input_file=input_path.name,
            created_at=now,
            updated_at=now,
        )

        return self._replace(
            record
        )

    def fail_interrupted_jobs(self) -> int:
        """Close orphaned queued/running jobs after a process restart.

        This app runs one local worker process.  Previous queued/running
        records have no in-memory future after an unclean restart; leaving
        them active would cause clients to poll indefinitely.  Completed and
        failed records, along with all generated files, are preserved.
        """
        recovered = 0
        with self._lock:
            if not self.root.is_dir():
                return 0
            for job_path in self.root.iterdir():
                if not job_path.is_dir() or not _JOB_ID_PATTERN.fullmatch(job_path.name):
                    continue
                try:
                    record = self.get(job_path.name)
                except (JobNotFoundError, OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                    continue  # Malformed unrelated metadata must not block startup.
                if record.status in {"queued", "running"}:
                    self.mark_failed(
                        record.job_id,
                        RuntimeError("服务器重启中断了任务；请重新提交图片。"),
                    )
                    recovered += 1
        return recovered

    def mark_running(
        self,
        job_id: str,
    ) -> JobRecord:
        current = self.get(
            job_id
        )
        now = _utc_now()

        return self._replace(
            replace(
                current,
                status="running",
                updated_at=now,
                started_at=(
                    current.started_at
                    or now
                ),
                completed_at=None,
                error=None,
                result_files={},
            )
        )

    def mark_completed(
        self,
        job_id: str,
        *,
        result_files: dict[str, str],
    ) -> JobRecord:
        current = self.get(
            job_id
        )
        now = _utc_now()

        return self._replace(
            replace(
                current,
                status="completed",
                updated_at=now,
                completed_at=now,
                error=None,
                result_files=dict(
                    result_files
                ),
            )
        )

    def mark_failed(
        self,
        job_id: str,
        error: BaseException,
    ) -> JobRecord:
        current = self.get(
            job_id
        )
        now = _utc_now()

        return self._replace(
            replace(
                current,
                status="failed",
                updated_at=now,
                completed_at=now,
                error={
                    "type":
                        type(error).__name__,
                    "message":
                        str(error),
                },
                result_files={},
            )
        )

    def manifest_path(
        self,
        job_id: str,
    ) -> Path:
        record = self.get(
            job_id
        )

        filename = (
            record.result_files
            .get(
                "manifest"
            )
        )

        if not filename:
            raise FileNotFoundError(
                f"job {job_id} 尚无 manifest"
            )

        path = (
            self.job_dir(job_id)
            / filename
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"找不到 job manifest：{path}"
            )

        return path

    def public_file_path(
        self,
        job_id: str,
        filename: str,
    ) -> Path:
        record = self.get(
            job_id
        )

        if (
            not filename
            or Path(filename).name
            != filename
        ):
            raise FileNotFoundError(
                f"非法结果文件名：{filename}"
            )

        allowed = {
            record.input_file,
            *record.result_files.values(),
        }

        if filename not in allowed:
            raise FileNotFoundError(
                f"文件不属于 job 的公开结果：{filename}"
            )

        path = (
            self.job_dir(job_id)
            / filename
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"找不到 job 结果文件：{path}"
            )

        return path
