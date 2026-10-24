from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src"),
)

from robot_portrait.preprocessing import (
    load_portrait_image,
    preprocess_bgr,
    preprocess_portrait,
)


class FakeDetector:
    """用于构造确定性人脸检测测试。"""

    def __init__(
        self,
        faces,
    ):
        self.faces = faces

    def detectMultiScale(
        self,
        *args,
        **kwargs,
    ):
        return self.faces


class FailingDetector:
    """模拟 OpenCV 检测器异常。"""

    def detectMultiScale(
        self,
        *args,
        **kwargs,
    ):
        raise cv2.error(
            "synthetic detector failure"
        )


def test_preprocess_bgr_uses_largest_face_and_clamps_crop():
    image = np.zeros(
        (80, 120, 3),
        dtype=np.uint8,
    )

    detector = FakeDetector(
        [
            (80, 45, 30, 30),

            # 最大的人脸位于左上角，
            # 用于验证裁剪区域不会越界。
            (0, 0, 50, 55),
        ]
    )

    result = preprocess_bgr(
        image,
        size=64,
        detector=detector,
        detector_name="fake",
    )

    assert result.mode == "face"

    assert result.face_box == (
        0,
        0,
        50,
        55,
    )

    assert result.crop_box == (
        0,
        0,
        80,
        80,
    )

    assert result.original_size == (
        120,
        80,
    )

    assert result.output_size == (
        64,
        64,
    )

    assert result.image.shape == (
        64,
        64,
        3,
    )

    assert result.warnings == ()


def test_preprocess_bgr_center_fallback_without_face():
    image = np.zeros(
        (40, 60, 3),
        dtype=np.uint8,
    )

    result = preprocess_bgr(
        image,
        size=32,
        detector=FakeDetector([]),
        detector_name="fake",
    )

    assert result.mode == "center"

    assert result.face_box is None

    assert result.crop_box == (
        10,
        0,
        50,
        40,
    )

    assert result.image.shape == (
        32,
        32,
        3,
    )

    assert result.warnings == (
        "未检测到人脸，"
        "改用中心方形裁剪。",
    )


def test_preprocess_bgr_detector_failure_falls_back():
    image = np.zeros(
        (40, 60, 3),
        dtype=np.uint8,
    )

    result = preprocess_bgr(
        image,
        size=32,
        detector=FailingDetector(),
        detector_name="failing",
    )

    assert result.mode == "center"

    assert result.warnings == (
        "Haar 人脸检测失败，"
        "改用中心方形裁剪。",
    )


@pytest.mark.parametrize(
    "size",
    [
        0,
        -1,
    ],
)
def test_preprocess_bgr_rejects_non_positive_size(
    size,
):
    image = np.zeros(
        (20, 20, 3),
        dtype=np.uint8,
    )

    with pytest.raises(
        ValueError,
        match="size 必须是正整数",
    ):
        preprocess_bgr(
            image,
            size=size,
        )


def test_load_portrait_image_rejects_corrupt_file(
    tmp_path,
):
    path = (
        tmp_path
        / "损坏图片.png"
    )

    path.write_bytes(
        b"not an image"
    )

    with pytest.raises(
        ValueError,
        match="无法读取输入照片",
    ):
        load_portrait_image(
            path
        )


def test_preprocess_portrait_returns_json_serializable_metadata(
    tmp_path,
):
    source = (
        tmp_path
        / "中文 输入.png"
    )

    image = np.full(
        (80, 120, 3),
        255,
        dtype=np.uint8,
    )

    ok, encoded = cv2.imencode(
        ".png",
        image,
    )

    assert ok

    encoded.tofile(
        str(source)
    )

    result = preprocess_portrait(
        source,
        size=48,
    )

    metadata = result.to_metadata(
        input_path=source,
        output_path=(
            tmp_path
            / "输出.png"
        ),
    )

    assert (
        metadata["input"]
        .endswith(
            "中文 输入.png"
        )
    )

    assert (
        metadata["output"]
        .endswith(
            "输出.png"
        )
    )

    assert metadata["size"] == 48

    assert metadata["output_size"] == {
        "width": 48,
        "height": 48,
    }

    assert metadata["mode"] in {
        "face",
        "center",
    }

    assert isinstance(
        metadata["warnings"],
        list,
    )