from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
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

    def __post_init__(self) -> None:
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


class JobStore:
    """Filesystem-backed storage for web generation jobs."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

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
        job_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        input_path = job_dir / "input.png"

        write_image(
            input_path,
            image,
        )

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
            created_at=(
                datetime.now(
                    timezone.utc
                )
                .isoformat()
            ),
        )

        self._write_json_atomic(
            job_dir / "job.json",
            record.to_metadata(),
        )

        return record
