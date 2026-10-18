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
    APDRAWINGGAN_ALIGNMENT_SIZE,
    align_apdrawinggan_face,
    apdrawinggan_alignment_template,
    fit_similarity_transform,
    read_apdrawinggan_landmarks,
    write_apdrawinggan_landmarks,
)


def _apply(
    points,
    matrix,
):
    points = np.asarray(
        points,
        dtype=np.float64,
    )

    homogeneous = np.concatenate(
        [
            points,
            np.ones(
                (
                    points.shape[0],
                    1,
                ),
                dtype=np.float64,
            ),
        ],
        axis=1,
    )

    return (
        homogeneous
        @ np.asarray(
            matrix,
            dtype=np.float64,
        ).T
    )


def test_template_matches_official_face_align_512_formula():
    template = (
        apdrawinggan_alignment_template()
    )

    expected = np.array(
        [
            [
                201.14285714285714,
                246.85714285714286,
            ],
            [
                310.85714285714283,
                246.85714285714286,
            ],
            [
                256.0,
                311.77142857142854,
            ],
            [
                206.62857142857143,
                370.8342857142857,
            ],
            [
                305.37142857142857,
                370.8342857142857,
            ],
        ],
        dtype=np.float64,
    )

    assert (
        APDRAWINGGAN_ALIGNMENT_SIZE
        == 512
    )

    assert np.allclose(
        template,
        expected,
        atol=1e-12,
    )


def test_template_scales_with_output_size():
    template_512 = (
        apdrawinggan_alignment_template(
            512
        )
    )

    template_256 = (
        apdrawinggan_alignment_template(
            256
        )
    )

    assert np.allclose(
        template_256,
        template_512 / 2.0,
        atol=1e-12,
    )

    with pytest.raises(
        ValueError
    ):
        apdrawinggan_alignment_template(
            0
        )


def test_similarity_recovers_rotation_scale_and_translation():
    source = np.array(
        [
            [10.0, 20.0],
            [80.0, 25.0],
            [45.0, 60.0],
            [20.0, 100.0],
            [75.0, 105.0],
        ],
        dtype=np.float64,
    )

    angle = np.deg2rad(
        17.0
    )
    scale = 1.35

    rotation = np.array(
        [
            [
                np.cos(angle),
                -np.sin(angle),
            ],
            [
                np.sin(angle),
                np.cos(angle),
            ],
        ],
        dtype=np.float64,
    )

    translation = np.array(
        [
            31.0,
            -14.0,
        ],
        dtype=np.float64,
    )

    target = (
        scale
        * (
            rotation
            @ source.T
        ).T
        + translation
    )

    result = (
        fit_similarity_transform(
            source,
            target,
        )
    )

    predicted = (
        result.transform_points(
            source
        )
    )

    assert result.reflected is False

    assert result.scale == pytest.approx(
        scale,
        abs=1e-12,
    )

    assert (
        result.rotation_degrees
        == pytest.approx(
            17.0,
            abs=1e-10,
        )
    )

    assert result.rms_error < 1e-10

    assert np.allclose(
        predicted,
        target,
        atol=1e-10,
    )


def test_similarity_can_select_reflection_like_matlab_similarity():
    source = np.array(
        [
            [10.0, 20.0],
            [80.0, 25.0],
            [45.0, 60.0],
            [20.0, 100.0],
            [75.0, 105.0],
        ],
        dtype=np.float64,
    )

    reflected_linear = np.array(
        [
            [-1.4, 0.0],
            [0.0, 1.4],
        ],
        dtype=np.float64,
    )

    target = (
        source
        @ reflected_linear.T
        + np.array(
            [
                200.0,
                15.0,
            ]
        )
    )

    reflected = (
        fit_similarity_transform(
            source,
            target,
            allow_reflection=True,
        )
    )

    nonreflected = (
        fit_similarity_transform(
            source,
            target,
            allow_reflection=False,
        )
    )

    assert reflected.reflected is True
    assert reflected.rms_error < 1e-10

    assert (
        nonreflected.rms_error
        > 1.0
    )


def test_degenerate_landmarks_are_rejected():
    source = np.array(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
        ],
        dtype=np.float64,
    )

    target = (
        apdrawinggan_alignment_template()
    )

    with pytest.raises(
        ValueError,
        match="退化",
    ):
        fit_similarity_transform(
            source,
            target,
        )


def test_align_face_outputs_512_and_white_border():
    template = (
        apdrawinggan_alignment_template()
    )

    # Source landmarks are shifted down/right, so the fitted transform
    # translates source content up/left.  The uncovered lower/right
    # output border must use APDrawingGAN's official white fill.
    source = (
        template
        + np.array(
            [
                20.0,
                10.0,
            ]
        )
    )

    image = np.zeros(
        (
            512,
            512,
            3,
        ),
        dtype=np.uint8,
    )

    result = (
        align_apdrawinggan_face(
            image,
            source,
        )
    )

    assert (
        result.image.shape
        == (
            512,
            512,
            3,
        )
    )

    assert (
        result.image.dtype
        == np.uint8
    )

    assert np.array_equal(
        result.image[
            0,
            0,
        ],
        np.array(
            [
                0,
                0,
                0,
            ],
            dtype=np.uint8,
        ),
    )

    assert np.array_equal(
        result.image[
            -1,
            -1,
        ],
        np.array(
            [
                255,
                255,
                255,
            ],
            dtype=np.uint8,
        ),
    )

    assert np.allclose(
        np.asarray(
            result.points
        ),
        template,
        atol=1e-9,
    )


def test_read_write_landmarks_round_trip_unicode_path(
    tmp_path,
):
    path = (
        tmp_path
        / "中文"
        / "对齐五点.txt"
    )

    points = np.array(
        [
            [1.2, 2.7],
            [3.4, 4.4],
            [5.0, 6.0],
            [7.8, 8.1],
            [9.2, 10.9],
        ],
        dtype=np.float64,
    )

    write_apdrawinggan_landmarks(
        path,
        points,
    )

    loaded = (
        read_apdrawinggan_landmarks(
            path
        )
    )

    expected = np.array(
        [
            [1.0, 3.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [8.0, 8.0],
            [9.0, 11.0],
        ],
        dtype=np.float64,
    )

    assert np.array_equal(
        loaded,
        expected,
    )


def test_invalid_landmark_file_is_rejected(
    tmp_path,
):
    path = (
        tmp_path
        / "bad.txt"
    )

    path.write_text(
        "1 2\n3 4\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="landmark",
    ):
        read_apdrawinggan_landmarks(
            path
        )


def test_real_yunet_sample_alignment_is_near_identity():
    # Fixed values from the real OpenCV YuNet smoke test on
    # APDrawingGAN's WechatIMG253_2 aligned sample.
    source = np.array(
        [
            [
                198.95361328125,
                251.3959197998047,
            ],
            [
                307.7003173828125,
                247.9546661376953,
            ],
            [
                259.6257629394531,
                313.3740539550781,
            ],
            [
                211.07522583007812,
                363.71844482421875,
            ],
            [
                304.2379455566406,
                360.32763671875,
            ],
        ],
        dtype=np.float64,
    )

    result = (
        fit_similarity_transform(
            source,
            apdrawinggan_alignment_template(),
        )
    )

    assert result.reflected is False

    assert result.scale == pytest.approx(
        1.0681031009,
        abs=1e-6,
    )

    assert (
        result.rotation_degrees
        == pytest.approx(
            2.1089840141,
            abs=1e-6,
        )
    )

    assert (
        result.rms_error
        == pytest.approx(
            3.8092764641,
            abs=1e-6,
        )
    )

    rounded = np.rint(
        result.transform_points(
            source
        )
    ).astype(
        np.int64
    )

    assert np.array_equal(
        rounded,
        np.array(
            [
                [197, 247],
                [313, 248],
                [259, 316],
                [205, 368],
                [305, 368],
            ],
            dtype=np.int64,
        ),
    )
