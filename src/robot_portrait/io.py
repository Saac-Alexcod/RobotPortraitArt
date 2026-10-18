from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

_SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def read_image(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Read an image through a Unicode-path-safe OpenCV code path."""
    image_path = Path(path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    if not image_path.is_file():
        raise ValueError(f"Image path is not a file: {image_path}")

    try:
        encoded = np.fromfile(str(image_path), dtype=np.uint8)
    except OSError as exc:
        raise OSError(f"Failed to read image bytes: {image_path}") from exc

    if encoded.size == 0:
        raise ValueError(f"Image file is empty: {image_path}")

    image = cv2.imdecode(encoded, flags)

    if image is None:
        raise ValueError(f"Unsupported or corrupted image: {image_path}")

    return image


def write_image(
    path: str | Path,
    image: np.ndarray,
    params: Iterable[int] | None = None,
) -> Path:
    """Write an image through a Unicode-path-safe OpenCV code path."""
    output_path = Path(path)

    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image must be a non-empty numpy array")

    suffix = output_path.suffix.lower()

    if suffix not in _SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported output image extension '{output_path.suffix}'. "
            f"Supported: {', '.join(sorted(_SUPPORTED_SUFFIXES))}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    encode_params = list(params) if params is not None else []

    ok, encoded = cv2.imencode(suffix, image, encode_params)

    if not ok:
        raise OSError(
            f"OpenCV failed to encode image as {suffix}: {output_path}"
        )

    try:
        encoded.tofile(str(output_path))
    except OSError as exc:
        raise OSError(f"Failed to write image: {output_path}") from exc

    return output_path