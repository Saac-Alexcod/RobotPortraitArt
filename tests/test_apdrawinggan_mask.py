from __future__ import annotations

from pathlib import Path
import sys

import cv2
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
    HumanMaskResult,
    HumanSegConfig,
    HumanSegModelError,
    PPHUMANSEG_INPUT_SIZE,
    compare_binary_masks,
    draw_human_mask_preview,
    generate_pphumanseg_mask,
)


class FakeNet:
    def __init__(
        self,
        output,
    ) -> None:
        self.output = output
        self.input_blob = None

    def setInput(
        self,
        blob,
    ) -> None:
        self.input_blob = (
            np.asarray(
                blob
            ).copy()
        )

    def forward(
        self,
    ):
        return self.output


def _model_file(
    tmp_path: Path,
) -> Path:
    path = (
        tmp_path
        / "中文模型"
        / "human_segmentation_pphumanseg_2023mar.onnx"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_bytes(
        b"fake-onnx"
        * 512
    )

    return path


def _image(
    width: int = 512,
    height: int = 512,
) -> np.ndarray:
    image = np.zeros(
        (
            height,
            width,
            3,
        ),
        dtype=np.uint8,
    )

    image[:, :, 0] = 32
    image[:, :, 1] = 128
    image[:, :, 2] = 224

    return image


def _two_class_logits(
    *,
    height: int = 192,
    width: int = 192,
) -> np.ndarray:
    output = np.zeros(
        (
            1,
            2,
            height,
            width,
        ),
        dtype=np.float32,
    )

    output[
        0,
        0,
        :,
        : width // 2,
    ] = 4.0

    output[
        0,
        1,
        :,
        width // 2 :,
    ] = 5.0

    return output


def test_constants_and_default_config(
    tmp_path,
):
    model = _model_file(
        tmp_path
    )

    config = HumanSegConfig(
        model
    )

    assert (
        PPHUMANSEG_INPUT_SIZE
        == (
            192,
            192,
        )
    )

    assert (
        config.foreground_class
        == 1
    )

    assert (
        config.backend_id
        == cv2.dnn.DNN_BACKEND_OPENCV
    )

    assert (
        config.target_id
        == cv2.dnn.DNN_TARGET_CPU
    )


def test_config_rejects_negative_foreground_class(
    tmp_path,
):
    model = _model_file(
        tmp_path
    )

    with pytest.raises(
        ValueError,
        match="foreground_class",
    ):
        HumanSegConfig(
            model,
            foreground_class=-1,
        )


def test_missing_model_is_reported(
    tmp_path,
):
    with pytest.raises(
        FileNotFoundError,
        match="PPHumanSeg ONNX",
    ):
        generate_pphumanseg_mask(
            _image(),
            HumanSegConfig(
                tmp_path
                / "missing.onnx"
            ),
        )


def test_git_lfs_pointer_is_reported(
    tmp_path,
):
    model = (
        tmp_path
        / "humanseg.onnx"
    )

    model.write_bytes(
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:123\n"
        b"size 999\n"
    )

    with pytest.raises(
        HumanSegModelError,
        match="Git LFS pointer",
    ):
        generate_pphumanseg_mask(
            _image(),
            HumanSegConfig(
                model
            ),
        )


def test_inference_uses_model_buffer_and_official_preprocess(
    tmp_path,
):
    model = _model_file(
        tmp_path
    )

    net = FakeNet(
        _two_class_logits()
    )

    captured = {}

    def factory(
        model_buffer,
        config,
    ):
        captured[
            "buffer_size"
        ] = int(
            model_buffer.size
        )

        captured[
            "model_path"
        ] = config.model_path

        return net

    result = (
        generate_pphumanseg_mask(
            _image(
                width=320,
                height=240,
            ),
            HumanSegConfig(
                model
            ),
            net_factory=factory,
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
            "model_path"
        ]
        == model
    )

    assert (
        net.input_blob.shape
        == (
            1,
            3,
            192,
            192,
        )
    )

    assert (
        net.input_blob.dtype
        == np.float32
    )

    assert (
        float(
            net.input_blob.min()
        )
        >= -1.0
    )

    assert (
        float(
            net.input_blob.max()
        )
        <= 1.0
    )

    assert (
        result.mask.shape
        == (
            240,
            320,
        )
    )


def test_foreground_is_white_and_background_is_black(
    tmp_path,
):
    model = _model_file(
        tmp_path
    )

    result = (
        generate_pphumanseg_mask(
            _image(
                width=200,
                height=100,
            ),
            HumanSegConfig(
                model
            ),
            net_factory=lambda *_args: FakeNet(
                _two_class_logits()
            ),
        )
    )

    assert (
        result.mask.dtype
        == np.uint8
    )

    assert set(
        np.unique(
            result.mask
        ).tolist()
    ) == {
        0,
        255,
    }

    assert np.all(
        result.mask[
            :,
            :90,
        ]
        == 0
    )

    assert np.all(
        result.mask[
            :,
            110:,
        ]
        == 255
    )

    assert (
        result.foreground_ratio
        == pytest.approx(
            0.5,
            abs=0.01,
        )
    )


def test_malformed_model_output_is_rejected(
    tmp_path,
):
    model = _model_file(
        tmp_path
    )

    with pytest.raises(
        RuntimeError,
        match="输出形状",
    ):
        generate_pphumanseg_mask(
            _image(),
            HumanSegConfig(
                model
            ),
            net_factory=lambda *_args: FakeNet(
                np.zeros(
                    (
                        1,
                        1,
                        192,
                        192,
                    ),
                    dtype=np.float32,
                )
            ),
        )


def test_foreground_class_out_of_range_is_rejected(
    tmp_path,
):
    model = _model_file(
        tmp_path
    )

    with pytest.raises(
        RuntimeError,
        match="foreground_class",
    ):
        generate_pphumanseg_mask(
            _image(),
            HumanSegConfig(
                model,
                foreground_class=2,
            ),
            net_factory=lambda *_args: FakeNet(
                _two_class_logits()
            ),
        )


def test_mask_result_requires_binary_uint8():
    with pytest.raises(
        ValueError,
        match="0/255",
    ):
        HumanMaskResult(
            mask=np.array(
                [
                    [
                        0,
                        127,
                    ]
                ],
                dtype=np.uint8,
            ),
            foreground_ratio=0.5,
        )

    with pytest.raises(
        ValueError,
        match="uint8",
    ):
        HumanMaskResult(
            mask=np.zeros(
                (
                    2,
                    2,
                ),
                dtype=np.float32,
            ),
            foreground_ratio=0.0,
        )


def test_preview_preserves_original_image():
    image = _image(
        width=100,
        height=80,
    )

    before = image.copy()

    mask = np.zeros(
        (
            80,
            100,
        ),
        dtype=np.uint8,
    )

    mask[
        :,
        50:,
    ] = 255

    result = HumanMaskResult(
        mask=mask,
        foreground_ratio=0.5,
    )

    preview = (
        draw_human_mask_preview(
            image,
            result,
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

    assert np.array_equal(
        preview[
            :,
            50:,
        ],
        image[
            :,
            50:,
        ],
    )

    assert not np.array_equal(
        preview[
            :,
            :50,
        ],
        image[
            :,
            :50,
        ],
    )


def test_compare_binary_masks_known_values():
    predicted = np.array(
        [
            [255, 255],
            [0, 0],
        ],
        dtype=np.uint8,
    )

    reference = np.array(
        [
            [255, 0],
            [255, 0],
        ],
        dtype=np.uint8,
    )

    metrics = (
        compare_binary_masks(
            predicted,
            reference,
        )
    )

    assert (
        metrics[
            "iou"
        ]
        == pytest.approx(
            1.0 / 3.0
        )
    )

    assert (
        metrics[
            "dice"
        ]
        == pytest.approx(
            0.5
        )
    )

    assert (
        metrics[
            "pixel_accuracy"
        ]
        == pytest.approx(
            0.5
        )
    )

    assert (
        metrics[
            "predicted_foreground_ratio"
        ]
        == pytest.approx(
            0.5
        )
    )

    assert (
        metrics[
            "reference_foreground_ratio"
        ]
        == pytest.approx(
            0.5
        )
    )


def test_compare_binary_masks_empty_foregrounds():
    empty = np.zeros(
        (
            3,
            4,
        ),
        dtype=np.uint8,
    )

    metrics = (
        compare_binary_masks(
            empty,
            empty,
        )
    )

    assert (
        metrics[
            "iou"
        ]
        == 1.0
    )

    assert (
        metrics[
            "dice"
        ]
        == 1.0
    )

    assert (
        metrics[
            "pixel_accuracy"
        ]
        == 1.0
    )
def test_postprocess_resizes_logits_before_argmax(
    tmp_path,
):
    model = _model_file(
        tmp_path
    )

    # This deliberately creates a case where:
    #   resize(argmax(logits)) != argmax(resize(logits))
    # The official OpenCV Zoo implementation uses the latter.
    logits = np.zeros(
        (
            1,
            2,
            1,
            2,
        ),
        dtype=np.float32,
    )

    logits[
        0,
        0,
        0,
        0,
    ] = 10.0

    logits[
        0,
        1,
        0,
        1,
    ] = 1.0

    result = (
        generate_pphumanseg_mask(
            _image(
                width=8,
                height=1,
            ),
            HumanSegConfig(
                model
            ),
            net_factory=lambda *_args: FakeNet(
                logits
            ),
        )
    )

    # Bilinear logits interpolation keeps class 0 dominant until x=6.
    assert np.array_equal(
        result.mask[
            0
        ],
        np.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                255,
                255,
            ],
            dtype=np.uint8,
        ),
    )
