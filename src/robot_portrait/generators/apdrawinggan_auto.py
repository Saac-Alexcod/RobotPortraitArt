from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Callable

import numpy as np

from .base import (
    GeneratorOutput,
    GeneratorRequest,
    GeneratorStatus,
    PortraitGenerator,
)
from ..apdrawinggan_preprocess import (
    FivePointFace,
    HumanMaskResult,
    HumanSegConfig,
    YuNetConfig,
    align_apdrawinggan_face,
    detect_yunet_five_points,
    draw_five_point_preview,
    draw_human_mask_preview,
    generate_pphumanseg_mask,
    write_apdrawinggan_landmarks,
)
from ..io import write_image
from ..preprocessing import load_portrait_image


DetectionFunction = Callable[
    [np.ndarray, YuNetConfig],
    FivePointFace,
]

MaskFunction = Callable[
    [np.ndarray, HumanSegConfig],
    HumanMaskResult,
]


@dataclass(frozen=True, slots=True)
class APDrawingGANAutoPreprocessConfig:
    """Models and output layout for APDrawingGAN automatic preprocessing."""

    yunet_model: Path
    humanseg_model: Path
    output_size: int = 512
    output_subdir: str = "apdrawinggan_preprocess"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "yunet_model",
            Path(self.yunet_model),
        )
        object.__setattr__(
            self,
            "humanseg_model",
            Path(self.humanseg_model),
        )

        if int(self.output_size) != 512:
            raise ValueError(
                "APDrawingGAN 自动预处理当前固定输出 512x512"
            )

        subdir = Path(
            self.output_subdir
        )

        if (
            subdir.is_absolute()
            or len(subdir.parts) != 1
            or self.output_subdir
            in {
                "",
                ".",
                "..",
            }
        ):
            raise ValueError(
                "output_subdir 必须是单层相对目录名"
            )


class APDrawingGANAutoPreprocessGenerator(
    PortraitGenerator
):
    """Wrap an APDrawingGAN backend with optional automatic preprocessing.

    Manual compatibility:
        If both ``landmark`` and ``mask`` are supplied, the request is passed
        to the existing APDrawingGAN backend unchanged.

    Automatic mode:
        If neither is supplied, the wrapper uses the original input image:
        YuNet -> official APDrawingGAN similarity alignment -> PPHumanSeg.
        The aligned image, aligned landmarks, and generated mask are then sent
        to the existing backend with ``preprocess_transform=None`` so they are
        not transformed a second time.
    """

    name = "apdrawinggan"

    def __init__(
        self,
        backend: PortraitGenerator,
        config: APDrawingGANAutoPreprocessConfig,
        *,
        detector: DetectionFunction | None = None,
        mask_generator: MaskFunction | None = None,
    ) -> None:
        if backend.name != self.name:
            raise ValueError(
                "自动预处理 wrapper 只能包装 "
                "name='apdrawinggan' 的生成器"
            )

        self.backend = backend
        self.config = config
        self._detector = (
            detector
            or detect_yunet_five_points
        )
        self._mask_generator = (
            mask_generator
            or generate_pphumanseg_mask
        )

    def status(
        self,
    ) -> GeneratorStatus:
        backend_status = (
            self.backend.status()
        )

        details = dict(
            backend_status.details
        )

        auto_ready = (
            self.config
            .yunet_model
            .is_file()
            and self.config
            .humanseg_model
            .is_file()
        )

        details[
            "auto_preprocess"
        ] = {
            "available": auto_ready,
            "yunet_model": str(
                self.config.yunet_model
            ),
            "yunet_model_present": (
                self.config
                .yunet_model
                .is_file()
            ),
            "humanseg_model": str(
                self.config.humanseg_model
            ),
            "humanseg_model_present": (
                self.config
                .humanseg_model
                .is_file()
            ),
            "pipeline": [
                "opencv-yunet",
                "apdrawinggan-similarity-512",
                "opencv-zoo-pphumanseg-2023mar",
            ],
        }

        return GeneratorStatus(
            available=(
                backend_status.available
            ),
            reason=backend_status.reason,
            details=details,
        )

    @staticmethod
    def _manual_pair(
        request: GeneratorRequest,
    ) -> tuple[
        Path | None,
        Path | None,
    ]:
        landmark = (
            request
            .auxiliary_inputs
            .get(
                "landmark"
            )
        )

        mask = (
            request
            .auxiliary_inputs
            .get(
                "mask"
            )
        )

        return (
            Path(landmark)
            if landmark is not None
            else None,
            Path(mask)
            if mask is not None
            else None,
        )

    @staticmethod
    def _merge_output(
        output: GeneratorOutput,
        preprocess_metadata: dict[
            str,
            object,
        ],
    ) -> GeneratorOutput:
        metadata = dict(
            output.metadata
        )

        metadata[
            "apdrawinggan_preprocess"
        ] = preprocess_metadata

        return GeneratorOutput(
            image=output.image,
            metadata=metadata,
            warnings=output.warnings,
        )

    def _auto_generate(
        self,
        request: GeneratorRequest,
        preprocess_dir: Path,
    ) -> GeneratorOutput:
        image = load_portrait_image(
            request.input_path
        )

        detection = self._detector(
            image,
            YuNetConfig(
                model_path=(
                    self.config
                    .yunet_model
                ),
            ),
        )

        detected_landmark_path = (
            preprocess_dir
            / "detected_landmarks.txt"
        )

        landmark_preview_path = (
            preprocess_dir
            / "landmark_preview.png"
        )

        aligned_image_path = (
            preprocess_dir
            / "aligned.png"
        )

        aligned_landmark_path = (
            preprocess_dir
            / "aligned_landmarks.txt"
        )

        mask_path = (
            preprocess_dir
            / "mask.png"
        )

        mask_preview_path = (
            preprocess_dir
            / "mask_preview.png"
        )

        write_apdrawinggan_landmarks(
            detected_landmark_path,
            detection,
        )

        write_image(
            landmark_preview_path,
            draw_five_point_preview(
                image,
                detection,
            ),
        )

        aligned = (
            align_apdrawinggan_face(
                image,
                detection,
                output_size=(
                    self.config
                    .output_size
                ),
            )
        )

        write_image(
            aligned_image_path,
            aligned.image,
        )

        write_apdrawinggan_landmarks(
            aligned_landmark_path,
            aligned.points,
        )

        mask_result = (
            self._mask_generator(
                aligned.image,
                HumanSegConfig(
                    model_path=(
                        self.config
                        .humanseg_model
                    ),
                ),
            )
        )

        write_image(
            mask_path,
            mask_result.mask,
        )

        write_image(
            mask_preview_path,
            draw_human_mask_preview(
                aligned.image,
                mask_result,
            ),
        )

        backend_request = (
            GeneratorRequest(
                prepared_image=(
                    aligned.image
                ),
                input_path=(
                    request.input_path
                ),
                prepared_path=(
                    aligned_image_path
                ),
                output_dir=(
                    request.output_dir
                ),
                seed=request.seed,
                # The image/landmarks/mask are already in the same
                # official APDrawingGAN 512x512 coordinate system.
                preprocess_transform=None,
                auxiliary_inputs={
                    "landmark":
                        aligned_landmark_path,
                    "mask":
                        mask_path,
                },
            )
        )

        output = self.backend.generate(
            backend_request
        )

        metadata: dict[
            str,
            object,
        ] = {
            "mode": "auto",
            "pipeline": [
                "opencv-yunet",
                "apdrawinggan-similarity-512",
                "opencv-zoo-pphumanseg-2023mar",
            ],
            "models": {
                "yunet": str(
                    self.config
                    .yunet_model
                ),
                "pphumanseg": str(
                    self.config
                    .humanseg_model
                ),
            },
            "detection":
                detection.to_metadata(),
            "alignment":
                aligned.to_metadata(),
            "mask":
                mask_result.to_metadata(),
            "files": {
                "detected_landmarks": str(
                    detected_landmark_path
                ),
                "landmark_preview": str(
                    landmark_preview_path
                ),
                "aligned_image": str(
                    aligned_image_path
                ),
                "aligned_landmarks": str(
                    aligned_landmark_path
                ),
                "mask": str(
                    mask_path
                ),
                "mask_preview": str(
                    mask_preview_path
                ),
            },
        }

        return self._merge_output(
            output,
            metadata,
        )

    def generate(
        self,
        request: GeneratorRequest,
    ) -> GeneratorOutput:
        landmark, mask = (
            self._manual_pair(
                request
            )
        )

        preprocess_dir = (
            Path(
                request.output_dir
            )
            / self.config
            .output_subdir
        )

        # Reusing one output directory must not leave stale auto artifacts.
        shutil.rmtree(
            preprocess_dir,
            ignore_errors=True,
        )

        if (
            landmark is not None
            and mask is not None
        ):
            output = (
                self.backend
                .generate(
                    request
                )
            )

            return self._merge_output(
                output,
                {
                    "mode": "manual",
                    "files": {
                        "landmark":
                            str(landmark),
                        "mask":
                            str(mask),
                    },
                },
            )

        if (
            landmark is not None
            or mask is not None
        ):
            raise ValueError(
                "APDrawingGAN 手动覆盖需要同时提供 "
                "landmark 和 mask；"
                "两者都省略时才启用自动预处理"
            )

        preprocess_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        return self._auto_generate(
            request,
            preprocess_dir,
        )
