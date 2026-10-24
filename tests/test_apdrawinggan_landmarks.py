from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src"),
)


from robot_portrait.apdrawinggan_preprocess import (
    FaceNotDetectedError,
    FivePointFace,
    YuNetConfig,
    YuNetModelError,
    detect_yunet_five_points,
    draw_five_point_preview,
    write_apdrawinggan_landmarks,
)


class FakeDetector:
    def __init__(
        self,
        faces,
    ) -> None:
        self.faces = faces
        self.last_image_shape = None

    def detect(
        self,
        image,
    ):
        self.last_image_shape = (
            image.shape
        )
        return 1, self.faces


def _model_file(
    tmp_path: Path,
) -> Path:
    path = (
        tmp_path
        / "模型"
        / "face_detection_yunet_2023mar.onnx"
    )
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_bytes(
        b"fake-onnx-data"
        * 256
    )
    return path


def _image(
    width: int = 512,
    height: int = 512,
) -> np.ndarray:
    return np.zeros(
        (
            height,
            width,
            3,
        ),
        dtype=np.uint8,
    )


def _face_row(
    *,
    x=100.0,
    y=120.0,
    width=220.0,
    height=260.0,
    score=0.98,
):
    # Landmark order mirrors YuNet/APDrawingGAN:
    # right eye, left eye, nose, right mouth, left mouth.
    return np.array(
        [
            x,
            y,
            width,
            height,
            197.0,
            248.0,
            316.0,
            247.0,
            254.0,
            324.0,
            204.0,
            363.0,
            310.0,
            365.0,
            score,
        ],
        dtype=np.float32,
    )


def test_config_rejects_invalid_values(
    tmp_path,
):
    model = _model_file(
        tmp_path
    )

    with pytest.raises(
        ValueError
    ):
        YuNetConfig(
            model,
            score_threshold=0.0,
        )

    with pytest.raises(
        ValueError
    ):
        YuNetConfig(
            model,
            nms_threshold=1.1,
        )

    with pytest.raises(
        ValueError
    ):
        YuNetConfig(
            model,
            max_input_side=0,
        )


def test_missing_model_is_reported(
    tmp_path,
):
    config = YuNetConfig(
        tmp_path
        / "missing.onnx"
    )

    with pytest.raises(
        FileNotFoundError,
        match="YuNet ONNX",
    ):
        detect_yunet_five_points(
            _image(),
            config,
        )


def test_git_lfs_pointer_is_reported(
    tmp_path,
):
    model = (
        tmp_path
        / "yunet.onnx"
    )

    model.write_bytes(
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:123\n"
        b"size 999\n"
    )

    with pytest.raises(
        YuNetModelError,
        match="Git LFS pointer",
    ):
        detect_yunet_five_points(
            _image(),
            YuNetConfig(model),
        )


def test_detects_official_five_point_order_and_uses_model_buffer(
    tmp_path,
):
    model = _model_file(
        tmp_path
    )

    captured = {}

    def factory(
        model_buffer,
        input_size,
        config,
    ):
        captured[
            "buffer_size"
        ] = int(
            model_buffer.size
        )
        captured[
            "input_size"
        ] = input_size
        captured[
            "model_path"
        ] = config.model_path

        return FakeDetector(
            np.stack(
                [
                    _face_row()
                ]
            )
        )

    result = (
        detect_yunet_five_points(
            _image(),
            YuNetConfig(model),
            detector_factory=factory,
        )
    )

    assert (
        result.rounded_points()
        == (
            (197, 248),
            (316, 247),
            (254, 324),
            (204, 363),
            (310, 365),
        )
    )

    assert (
        captured[
            "buffer_size"
        ]
        == model.stat().st_size
    )

    assert (
        captured[
            "input_size"
        ]
        == (
            512,
            512,
        )
    )

    assert (
        captured[
            "model_path"
        ]
        == model
    )


def test_prefers_largest_face_over_smaller_higher_confidence(
    tmp_path,
):
    model = _model_file(
        tmp_path
    )

    small = _face_row(
        x=10,
        y=10,
        width=80,
        height=80,
        score=0.999,
    )

    large = _face_row(
        x=100,
        y=90,
        width=250,
        height=280,
        score=0.91,
    )

    def factory(
        *_args,
    ):
        return FakeDetector(
            np.stack(
                [
                    small,
                    large,
                ]
            )
        )

    result = (
        detect_yunet_five_points(
            _image(),
            YuNetConfig(model),
            detector_factory=factory,
        )
    )

    assert (
        result.face_box
        == (
            100,
            90,
            350,
            370,
        )
    )

    assert result.score == pytest.approx(
        0.91
    )


def test_large_image_is_scaled_for_detection_and_mapped_back(
    tmp_path,
):
    model = _model_file(
        tmp_path
    )

    detector = FakeDetector(
        np.array(
            [
                [
                    50,
                    60,
                    100,
                    120,
                    80,
                    100,
                    120,
                    100,
                    100,
                    130,
                    85,
                    155,
                    115,
                    155,
                    0.97,
                ]
            ],
            dtype=np.float32,
        )
    )

    def factory(
        _buffer,
        input_size,
        _config,
    ):
        assert (
            input_size
            == (
                600,
                400,
            )
        )
        return detector

    result = (
        detect_yunet_five_points(
            _image(
                width=1200,
                height=800,
            ),
            YuNetConfig(
                model,
                max_input_side=600,
            ),
            detector_factory=factory,
        )
    )

    assert (
        detector.last_image_shape
        == (
            400,
            600,
            3,
        )
    )

    assert (
        result.face_box
        == (
            100,
            120,
            300,
            360,
        )
    )

    assert (
        result.rounded_points()[0]
        == (
            160,
            200,
        )
    )


def test_no_face_is_reported(
    tmp_path,
):
    model = _model_file(
        tmp_path
    )

    def factory(
        *_args,
    ):
        return FakeDetector(
            None
        )

    with pytest.raises(
        FaceNotDetectedError,
        match="未检测到人脸",
    ):
        detect_yunet_five_points(
            _image(),
            YuNetConfig(model),
            detector_factory=factory,
        )


def test_malformed_detector_output_is_rejected(
    tmp_path,
):
    model = _model_file(
        tmp_path
    )

    def factory(
        *_args,
    ):
        return FakeDetector(
            np.zeros(
                (
                    1,
                    10,
                ),
                dtype=np.float32,
            )
        )

    with pytest.raises(
        RuntimeError,
        match="检测结果形状",
    ):
        detect_yunet_five_points(
            _image(),
            YuNetConfig(model),
            detector_factory=factory,
        )


def test_invalid_image_is_rejected(
    tmp_path,
):
    model = _model_file(
        tmp_path
    )

    with pytest.raises(
        ValueError,
        match="HxWx3",
    ):
        detect_yunet_five_points(
            np.zeros(
                (
                    32,
                    32,
                ),
                dtype=np.uint8,
            ),
            YuNetConfig(model),
        )

    with pytest.raises(
        ValueError,
        match="uint8",
    ):
        detect_yunet_five_points(
            np.zeros(
                (
                    32,
                    32,
                    3,
                ),
                dtype=np.float32,
            ),
            YuNetConfig(model),
        )


def test_writer_supports_unicode_path_and_rounds_points(
    tmp_path,
):
    output = (
        tmp_path
        / "中文输出"
        / "五点.txt"
    )

    points = np.array(
        [
            [10.2, 20.7],
            [30.6, 40.2],
            [50.0, 60.0],
            [70.49, 80.49],
            [90.51, 100.51],
        ],
        dtype=np.float64,
    )

    write_apdrawinggan_landmarks(
        output,
        points,
    )

    assert output.read_text(
        encoding="utf-8"
    ) == (
        "10 21\n"
        "31 40\n"
        "50 60\n"
        "70 80\n"
        "91 101\n"
    )


def test_preview_preserves_input_and_dimensions():
    image = _image(
        width=300,
        height=200,
    )

    before = image.copy()

    detection = FivePointFace(
        points=(
            (100.0, 80.0),
            (150.0, 80.0),
            (125.0, 105.0),
            (105.0, 130.0),
            (145.0, 130.0),
        ),
        face_box=(
            70,
            40,
            180,
            160,
        ),
        score=0.99,
        original_size=(
            300,
            200,
        ),
    )

    preview = (
        draw_five_point_preview(
            image,
            detection,
        )
    )

    assert (
        preview.shape
        == image.shape
    )

    assert np.array_equal(
        image,
        before,
    )

    assert not np.array_equal(
        preview,
        image,
    )


def test_metadata_contains_named_landmarks():
    detection = FivePointFace(
        points=(
            (1.0, 2.0),
            (3.0, 4.0),
            (5.0, 6.0),
            (7.0, 8.0),
            (9.0, 10.0),
        ),
        face_box=(
            0,
            0,
            20,
            20,
        ),
        score=0.95,
        original_size=(
            20,
            20,
        ),
    )

    metadata = (
        detection.to_metadata()
    )

    assert (
        metadata[
            "landmarks"
        ][
            "right_eye"
        ]
        == {
            "x": 1.0,
            "y": 2.0,
        }
    )

    assert (
        metadata[
            "landmarks"
        ][
            "left_mouth_corner"
        ]
        == {
            "x": 9.0,
            "y": 10.0,
        }
    )
