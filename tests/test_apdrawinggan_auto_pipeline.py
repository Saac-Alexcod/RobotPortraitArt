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
    FivePointFace,
    HumanMaskResult,
    apdrawinggan_alignment_template,
)
from robot_portrait.generators.apdrawinggan_auto import (
    APDrawingGANAutoPreprocessConfig,
    APDrawingGANAutoPreprocessGenerator,
)
from robot_portrait.generators.base import (
    GeneratorOutput,
    GeneratorRequest,
    GeneratorStatus,
    PortraitGenerator,
)
from robot_portrait.io import (
    read_image,
    write_image,
)


class CapturingBackend(
    PortraitGenerator
):
    name = "apdrawinggan"

    def __init__(
        self,
        *,
        available: bool = True,
        warnings: tuple[str, ...] = (),
    ) -> None:
        self.available = available
        self.warnings = warnings
        self.requests = []

    def status(
        self,
    ) -> GeneratorStatus:
        return GeneratorStatus(
            available=self.available,
            reason=(
                None
                if self.available
                else "backend unavailable"
            ),
            details={
                "backend":
                    "fake-apdrawinggan"
            },
        )

    def generate(
        self,
        request: GeneratorRequest,
    ) -> GeneratorOutput:
        self.requests.append(
            request
        )

        return GeneratorOutput(
            image=np.full(
                (
                    512,
                    512,
                    3,
                ),
                240,
                dtype=np.uint8,
            ),
            metadata={
                "backend":
                    "apdrawinggan"
            },
            warnings=self.warnings,
        )


def _source_image(
    width: int = 320,
    height: int = 240,
) -> np.ndarray:
    image = np.zeros(
        (
            height,
            width,
            3,
        ),
        dtype=np.uint8,
    )
    image[
        :,
        :,
        0,
    ] = 17
    image[
        :,
        :,
        1,
    ] = 91
    image[
        :,
        :,
        2,
    ] = 203

    return image


def _write_source(
    tmp_path: Path,
) -> Path:
    path = (
        tmp_path
        / "中文输入"
        / "portrait.png"
    )

    write_image(
        path,
        _source_image(),
    )

    return path


def _request(
    tmp_path: Path,
    input_path: Path,
    *,
    auxiliary_inputs=None,
) -> GeneratorRequest:
    # Deliberately different from the source image.  Auto preprocessing must
    # use input_path, not this generic shared-preprocess image.
    prepared = np.full(
        (
            512,
            512,
            3,
        ),
        7,
        dtype=np.uint8,
    )

    return GeneratorRequest(
        prepared_image=prepared,
        input_path=input_path,
        prepared_path=(
            tmp_path
            / "prepared.png"
        ),
        output_dir=(
            tmp_path
            / "输出"
        ),
        seed=42,
        auxiliary_inputs=(
            auxiliary_inputs
            or {}
        ),
    )


def _detector(
    seen: dict,
):
    def detect(
        image,
        config,
    ):
        seen[
            "shape"
        ] = tuple(
            image.shape
        )

        seen[
            "mean"
        ] = float(
            image.mean()
        )

        seen[
            "model"
        ] = config.model_path

        template = (
            apdrawinggan_alignment_template()
        )

        # Map the 512 template into a compact coordinate system that lies
        # inside the 320x240 synthetic input.  Official alignment should
        # recover the template again.
        points = (
            template
            * 0.40
            + np.array(
                [
                    25.0,
                    5.0,
                ]
            )
        )

        return FivePointFace(
            points=tuple(
                (
                    float(x),
                    float(y),
                )
                for x, y
                in points
            ),
            face_box=(
                70,
                75,
                180,
                220,
            ),
            score=0.97,
            original_size=(
                image.shape[1],
                image.shape[0],
            ),
        )

    return detect


def _mask_generator(
    seen: dict,
):
    def generate(
        image,
        config,
    ):
        seen[
            "mask_shape"
        ] = tuple(
            image.shape
        )

        seen[
            "humanseg_model"
        ] = config.model_path

        mask = np.zeros(
            image.shape[:2],
            dtype=np.uint8,
        )

        mask[
            80:500,
            90:430,
        ] = 255

        return HumanMaskResult(
            mask=mask,
            foreground_ratio=float(
                np.count_nonzero(
                    mask
                )
                / mask.size
            ),
        )

    return generate


def _wrapper(
    tmp_path: Path,
    backend: PortraitGenerator,
    *,
    detector=None,
    mask_generator=None,
):
    return APDrawingGANAutoPreprocessGenerator(
        backend,
        APDrawingGANAutoPreprocessConfig(
            yunet_model=(
                tmp_path
                / "models"
                / "yunet.onnx"
            ),
            humanseg_model=(
                tmp_path
                / "models"
                / "humanseg.onnx"
            ),
        ),
        detector=detector,
        mask_generator=mask_generator,
    )


def test_auto_mode_uses_original_image_and_passes_aligned_inputs(
    tmp_path,
):
    source_path = (
        _write_source(
            tmp_path
        )
    )

    seen = {}
    backend = (
        CapturingBackend()
    )

    wrapper = _wrapper(
        tmp_path,
        backend,
        detector=_detector(
            seen
        ),
        mask_generator=(
            _mask_generator(
                seen
            )
        ),
    )

    output = wrapper.generate(
        _request(
            tmp_path,
            source_path,
        )
    )

    assert seen[
        "shape"
    ] == (
        240,
        320,
        3,
    )

    assert seen[
        "mask_shape"
    ] == (
        512,
        512,
        3,
    )

    assert len(
        backend.requests
    ) == 1

    delegated = (
        backend.requests[0]
    )

    assert (
        delegated
        .preprocess_transform
        is None
    )

    assert (
        delegated
        .prepared_image
        .shape
        == (
            512,
            512,
            3,
        )
    )

    assert set(
        delegated
        .auxiliary_inputs
    ) == {
        "landmark",
        "mask",
    }

    assert (
        output.metadata[
            "apdrawinggan_preprocess"
        ][
            "mode"
        ]
        == "auto"
    )


def test_auto_mode_writes_explainable_intermediate_artifacts(
    tmp_path,
):
    source_path = (
        _write_source(
            tmp_path
        )
    )

    backend = (
        CapturingBackend()
    )

    wrapper = _wrapper(
        tmp_path,
        backend,
        detector=_detector(
            {}
        ),
        mask_generator=(
            _mask_generator(
                {}
            )
        ),
    )

    output = wrapper.generate(
        _request(
            tmp_path,
            source_path,
        )
    )

    metadata = output.metadata[
        "apdrawinggan_preprocess"
    ]

    files = metadata[
        "files"
    ]

    expected_names = {
        "detected_landmarks":
            "detected_landmarks.txt",
        "landmark_preview":
            "landmark_preview.png",
        "aligned_image":
            "aligned.png",
        "aligned_landmarks":
            "aligned_landmarks.txt",
        "mask":
            "mask.png",
        "mask_preview":
            "mask_preview.png",
    }

    for key, name in (
        expected_names.items()
    ):
        path = Path(
            files[key]
        )
        assert path.name == name
        assert path.is_file()

    mask = read_image(
        files[
            "mask"
        ],
        flags=0,
    )

    assert set(
        np.unique(
            mask
        ).tolist()
    ) == {
        0,
        255,
    }


def test_auto_landmarks_are_aligned_to_official_template(
    tmp_path,
):
    source_path = (
        _write_source(
            tmp_path
        )
    )

    backend = (
        CapturingBackend()
    )

    wrapper = _wrapper(
        tmp_path,
        backend,
        detector=_detector(
            {}
        ),
        mask_generator=(
            _mask_generator(
                {}
            )
        ),
    )

    output = wrapper.generate(
        _request(
            tmp_path,
            source_path,
        )
    )

    landmark_path = Path(
        output.metadata[
            "apdrawinggan_preprocess"
        ][
            "files"
        ][
            "aligned_landmarks"
        ]
    )

    points = np.loadtxt(
        landmark_path,
        dtype=np.float64,
    )

    expected = np.rint(
        apdrawinggan_alignment_template()
    )

    assert np.array_equal(
        points,
        expected,
    )


def test_manual_pair_preserves_existing_backend_behavior(
    tmp_path,
):
    source_path = (
        _write_source(
            tmp_path
        )
    )

    landmark = (
        tmp_path
        / "manual.txt"
    )

    landmark.write_text(
        (
            "1 1\n"
            "2 2\n"
            "3 3\n"
            "4 4\n"
            "5 5\n"
        ),
        encoding="utf-8",
    )

    mask = (
        tmp_path
        / "manual.png"
    )

    write_image(
        mask,
        np.zeros(
            (
                240,
                320,
            ),
            dtype=np.uint8,
        ),
    )

    backend = (
        CapturingBackend()
    )

    wrapper = _wrapper(
        tmp_path,
        backend,
    )

    request = _request(
        tmp_path,
        source_path,
        auxiliary_inputs={
            "landmark":
                landmark,
            "mask":
                mask,
        },
    )

    output = wrapper.generate(
        request
    )

    assert (
        backend.requests[0]
        is request
    )

    assert (
        output.metadata[
            "apdrawinggan_preprocess"
        ][
            "mode"
        ]
        == "manual"
    )


def test_partial_manual_override_is_rejected(
    tmp_path,
):
    source_path = (
        _write_source(
            tmp_path
        )
    )

    landmark = (
        tmp_path
        / "manual.txt"
    )

    landmark.write_text(
        "1 1\n",
        encoding="utf-8",
    )

    wrapper = _wrapper(
        tmp_path,
        CapturingBackend(),
    )

    with pytest.raises(
        ValueError,
        match="同时提供",
    ):
        wrapper.generate(
            _request(
                tmp_path,
                source_path,
                auxiliary_inputs={
                    "landmark":
                        landmark,
                },
            )
        )


def test_status_reports_auto_models_without_breaking_manual_availability(
    tmp_path,
):
    backend = (
        CapturingBackend(
            available=True
        )
    )

    wrapper = _wrapper(
        tmp_path,
        backend,
    )

    status = wrapper.status()

    assert status.available is True

    auto = status.details[
        "auto_preprocess"
    ]

    assert (
        auto[
            "available"
        ]
        is False
    )

    assert (
        auto[
            "yunet_model_present"
        ]
        is False
    )

    assert (
        auto[
            "humanseg_model_present"
        ]
        is False
    )


def test_backend_unavailability_is_preserved(
    tmp_path,
):
    wrapper = _wrapper(
        tmp_path,
        CapturingBackend(
            available=False
        ),
    )

    status = wrapper.status()

    assert status.available is False

    assert (
        status.reason
        == "backend unavailable"
    )


def test_repeated_run_removes_stale_auto_artifacts_and_preserves_warnings(
    tmp_path,
):
    source_path = (
        _write_source(
            tmp_path
        )
    )

    backend = (
        CapturingBackend(
            warnings=(
                "backend warning",
            )
        )
    )

    wrapper = _wrapper(
        tmp_path,
        backend,
        detector=_detector(
            {}
        ),
        mask_generator=(
            _mask_generator(
                {}
            )
        ),
    )

    request = _request(
        tmp_path,
        source_path,
    )

    stale_dir = (
        request.output_dir
        / "apdrawinggan_preprocess"
    )

    stale_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stale = (
        stale_dir
        / "stale.txt"
    )

    stale.write_text(
        "old",
        encoding="utf-8",
    )

    output = wrapper.generate(
        request
    )

    assert not stale.exists()

    assert (
        output.warnings
        == (
            "backend warning",
        )
    )


def test_config_rejects_non_512_and_unsafe_subdir(
    tmp_path,
):
    with pytest.raises(
        ValueError,
        match="512x512",
    ):
        APDrawingGANAutoPreprocessConfig(
            yunet_model=(
                tmp_path
                / "yunet.onnx"
            ),
            humanseg_model=(
                tmp_path
                / "humanseg.onnx"
            ),
            output_size=256,
        )

    with pytest.raises(
        ValueError,
        match="单层相对目录名",
    ):
        APDrawingGANAutoPreprocessConfig(
            yunet_model=(
                tmp_path
                / "yunet.onnx"
            ),
            humanseg_model=(
                tmp_path
                / "humanseg.onnx"
            ),
            output_subdir=(
                "../outside"
            ),
        )
