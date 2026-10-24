"""Verify public example images, including on Windows Unicode paths."""

from pathlib import Path

import cv2
import numpy as np
import pytest


EXAMPLE_FILES = (
    "portrait_sample.jpg",
    "portrait_long_hair.jpg",
    "portrait_glasses.jpg",
    "portrait_profile_challenge.jpg",
)


@pytest.mark.parametrize("filename", EXAMPLE_FILES)
def test_public_portrait_sample_is_decodable(filename: str) -> None:
    sample = Path(__file__).resolve().parents[1] / "data" / "examples" / filename
    assert sample.is_file(), f"Missing public example: {sample}"

    # cv2.imread(path) can fail for Windows paths containing Chinese characters.
    encoded = np.frombuffer(sample.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    assert image is not None, f"Cannot decode: {sample}"
    assert image.shape[0] >= 256 and image.shape[1] >= 256
    assert image.ndim == 3 and image.shape[2] == 3


@pytest.mark.parametrize("filename", EXAMPLE_FILES)
def test_frontend_gallery_asset_matches_public_example(filename: str) -> None:
    """The web gallery and CLI must ship the same example bytes."""
    repository = Path(__file__).resolve().parents[1]
    original = repository / "data" / "examples" / filename
    web_asset = repository / "frontend" / "public" / "examples" / filename
    assert web_asset.is_file(), f"Missing Vite gallery asset: {web_asset}"
    assert web_asset.read_bytes() == original.read_bytes(), f"Stale gallery copy: {filename}"
