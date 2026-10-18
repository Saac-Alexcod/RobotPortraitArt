from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
import shutil
import subprocess
from typing import Callable, Sequence

import cv2
import numpy as np

from .apdrawinggan import (
    APDrawingGANPreflight,
    APDrawingGANPreflightConfig,
    APDrawingGANPreflightReport,
)
from .base import (
    GeneratorOutput,
    GeneratorRequest,
    GeneratorStatus,
    PortraitGenerator,
    PreprocessTransform,
)
from ..io import read_image, write_image


GeneratorCommandRunner = Callable[
    [Sequence[str]],
    subprocess.CompletedProcess[str],
]
WSLPathResolver = Callable[[Path], str]
PreflightProvider = Callable[
    [],
    APDrawingGANPreflightReport,
]


@dataclass(frozen=True, slots=True)
class APDrawingGANGeneratorConfig:
    """Runtime configuration for APDrawingGAN inference through WSL."""

    repo: Path
    distro: str = "Ubuntu-22.04"
    python_command: str = "python3"
    experiment: str = "formal_author"
    epoch: str = "300"
    gpu_ids: str = "0"
    wsl_executable: str = "wsl.exe"
    require_cuda: bool = True
    timeout_seconds: float = 180.0
    keep_work_dir: bool = False
    fine_size: int = 512

    def __post_init__(self) -> None:
        if not str(self.python_command).strip():
            raise ValueError(
                "python_command 不能为空"
            )
        if not str(self.experiment).strip():
            raise ValueError(
                "experiment 不能为空"
            )
        if not str(self.epoch).strip():
            raise ValueError(
                "epoch 不能为空"
            )
        if not str(self.gpu_ids).strip():
            raise ValueError(
                "gpu_ids 不能为空"
            )
        if float(self.timeout_seconds) <= 0:
            raise ValueError(
                "timeout_seconds 必须为正数"
            )
        if int(self.fine_size) <= 0:
            raise ValueError(
                "fine_size 必须为正整数"
            )


class APDrawingGANGenerator(PortraitGenerator):
    """Run the official APDrawingGAN test pipeline inside WSL.

    The generator consumes the pipeline-prepared 512x512 BGR image plus
    two explicit auxiliary inputs defined in the original-image coordinate
    system:

    - ``landmark``: official five-point ``.txt`` format.
    - ``mask``: official background-mask ``.png`` format.

    When the main pipeline crops/resizes the source portrait, the same
    transform is applied to the landmarks and mask before invoking the
    legacy APDrawingGAN code.
    """

    name = "apdrawinggan"

    def __init__(
        self,
        config: APDrawingGANGeneratorConfig,
        *,
        runner: GeneratorCommandRunner | None = None,
        path_resolver: WSLPathResolver | None = None,
        preflight_provider: PreflightProvider | None = None,
    ) -> None:
        self.config = config
        self._runner = (
            runner
            or self._default_runner
        )
        self._path_resolver = (
            path_resolver
            or self._resolve_wsl_path
        )
        self._preflight_provider = (
            preflight_provider
            or self._run_preflight
        )

    def _default_runner(
        self,
        command: Sequence[str],
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.config.timeout_seconds,
        )

    def _run_preflight(
        self,
    ) -> APDrawingGANPreflightReport:
        return APDrawingGANPreflight(
            APDrawingGANPreflightConfig(
                repo=self.config.repo,
                distro=self.config.distro,
                python_command=(
                    self.config.python_command
                ),
                experiment=(
                    self.config.experiment
                ),
                epoch=self.config.epoch,
                wsl_executable=(
                    self.config.wsl_executable
                ),
                require_cuda=(
                    self.config.require_cuda
                ),
            )
        ).run()

    @staticmethod
    def _combined_output(
        completed: subprocess.CompletedProcess[str],
    ) -> str:
        return "\n".join(
            part.strip()
            for part in (
                completed.stdout or "",
                completed.stderr or "",
            )
            if part.strip()
        ).strip()

    @staticmethod
    def _translate_windows_drive_path(
        path: str | Path,
    ) -> str | None:
        """Translate ``C:\\...`` into WSL's default ``/mnt/c/...``.

        This intentionally avoids invoking ``wslpath`` for normal drive-letter
        paths.  ``wsl.exe`` can consume backslashes while forwarding a Windows
        path as a Linux command argument, which breaks paths containing spaces
        or non-ASCII characters.
        """

        windows_path = (
            PureWindowsPath(
                str(path)
            )
        )
        drive = (
            windows_path.drive
        )

        if not (
            len(drive) == 2
            and drive[1] == ":"
            and drive[0].isalpha()
        ):
            return None

        mount = (
            "/mnt/"
            + drive[0].lower()
        )
        tail = "/".join(
            windows_path.parts[1:]
        )

        return (
            f"{mount}/{tail}"
            if tail
            else mount
        )

    def _resolve_wsl_path(
        self,
        path: Path,
    ) -> str:
        resolved = (
            path.resolve()
        )

        direct = (
            self
            ._translate_windows_drive_path(
                resolved
            )
        )

        if direct is not None:
            return direct

        completed = self._runner(
            [
                self.config.wsl_executable,
                "-d",
                self.config.distro,
                "--cd",
                "~",
                "--",
                "wslpath",
                "-a",
                "-u",
                str(resolved),
            ]
        )

        converted = (
            completed.stdout
            or ""
        ).strip()

        if (
            completed.returncode != 0
            or not converted
        ):
            detail = (
                self._combined_output(
                    completed
                )
            )
            suffix = (
                f"\n{detail}"
                if detail
                else ""
            )

            raise RuntimeError(
                "无法将路径转换为 WSL 路径："
                f"{path}{suffix}"
            )

        return converted

    def _wsl_repo_visible(
        self,
        repo_wsl: str,
    ) -> tuple[bool, str]:
        completed = self._runner(
            [
                self.config.wsl_executable,
                "-d",
                self.config.distro,
                "--cd",
                "~",
                "--",
                "test",
                "-f",
                f"{repo_wsl}/test.py",
            ]
        )
        return (
            completed.returncode == 0,
            self._combined_output(
                completed
            ),
        )

    def status(
        self,
    ) -> GeneratorStatus:
        report = (
            self._preflight_provider()
        )

        if not report.ready:
            reason = "; ".join(
                check.message
                for check
                in report
                .failed_required_checks()
                if check.message
            )
            return GeneratorStatus(
                available=False,
                reason=(
                    reason
                    or "APDrawingGAN preflight 未通过"
                ),
                details={
                    "backend":
                        "apdrawinggan",
                    "preflight":
                        report.to_metadata(),
                },
            )

        try:
            repo_wsl = (
                self._path_resolver(
                    self.config.repo
                )
            )
            visible, probe_output = (
                self._wsl_repo_visible(
                    repo_wsl
                )
            )
        except Exception as exc:
            return GeneratorStatus(
                available=False,
                reason=str(exc),
                details={
                    "backend":
                        "apdrawinggan",
                    "preflight":
                        report.to_metadata(),
                },
            )

        if not visible:
            return GeneratorStatus(
                available=False,
                reason=(
                    "WSL 无法访问 APDrawingGAN 源码目录，"
                    "请确认对应 Windows 盘已挂载。"
                ),
                details={
                    "backend":
                        "apdrawinggan",
                    "repo":
                        str(self.config.repo),
                    "repo_wsl":
                        repo_wsl,
                    "probe_output":
                        probe_output,
                    "preflight":
                        report.to_metadata(),
                },
            )

        return GeneratorStatus(
            available=True,
            details={
                "backend":
                    "apdrawinggan",
                "repo":
                    str(self.config.repo),
                "repo_wsl":
                    repo_wsl,
                "distro":
                    self.config.distro,
                "python":
                    self.config.python_command,
                "experiment":
                    self.config.experiment,
                "epoch":
                    self.config.epoch,
                "gpu_ids":
                    self.config.gpu_ids,
                "preflight":
                    report.to_metadata(),
            },
        )

    def _validate_request(
        self,
        request: GeneratorRequest,
    ) -> tuple[Path, Path]:
        height, width = (
            request
            .prepared_image
            .shape[:2]
        )
        expected = (
            self.config.fine_size
        )

        if (
            width,
            height,
        ) != (
            expected,
            expected,
        ):
            raise ValueError(
                "APDrawingGAN 当前要求 prepared_image 为 "
                f"{expected}x{expected}，"
                f"实际为 {width}x{height}"
            )

        try:
            landmark = Path(
                request
                .auxiliary_inputs[
                    "landmark"
                ]
            )
            mask = Path(
                request
                .auxiliary_inputs[
                    "mask"
                ]
            )
        except KeyError as exc:
            raise ValueError(
                "APDrawingGAN 需要 auxiliary_inputs "
                "中同时提供 landmark 和 mask"
            ) from exc

        if not landmark.is_file():
            raise FileNotFoundError(
                "找不到 APDrawingGAN landmark："
                f"{landmark}"
            )
        if not mask.is_file():
            raise FileNotFoundError(
                "找不到 APDrawingGAN mask："
                f"{mask}"
            )

        return (
            landmark,
            mask,
        )

    @staticmethod
    def _load_landmarks(
        path: Path,
    ) -> np.ndarray:
        try:
            points = np.loadtxt(
                path,
                dtype=np.float64,
            )
        except Exception as exc:
            raise ValueError(
                "无法读取 APDrawingGAN landmark："
                f"{path}"
            ) from exc

        points = np.asarray(
            points,
            dtype=np.float64,
        )

        if (
            points.shape != (5, 2)
            or not np.isfinite(
                points
            ).all()
        ):
            raise ValueError(
                "APDrawingGAN landmark 必须恰好包含 "
                "5 个有限的 (x, y) 点"
            )

        return points

    @staticmethod
    def _transform_landmarks(
        points: np.ndarray,
        transform: (
            PreprocessTransform
            | None
        ),
    ) -> np.ndarray:
        if transform is None:
            return np.rint(
                points
            ).astype(
                np.int64
            )

        left, top, right, bottom = (
            transform.crop_box
        )
        output_width, output_height = (
            transform.output_size
        )
        scale_x = (
            output_width
            / (right - left)
        )
        scale_y = (
            output_height
            / (bottom - top)
        )

        transformed = points.copy()
        transformed[:, 0] = (
            transformed[:, 0]
            - left
        ) * scale_x
        transformed[:, 1] = (
            transformed[:, 1]
            - top
        ) * scale_y

        return np.rint(
            transformed
        ).astype(
            np.int64
        )

    @staticmethod
    def _write_landmarks(
        path: Path,
        points: np.ndarray,
    ) -> None:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        path.write_text(
            "".join(
                f"{int(x)} {int(y)}\n"
                for x, y
                in points
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _prepare_mask(
        source: Path,
        destination: Path,
        transform: (
            PreprocessTransform
            | None
        ),
        expected_size: int,
    ) -> None:
        mask = read_image(
            source,
            cv2.IMREAD_GRAYSCALE,
        )

        if transform is not None:
            expected_shape = (
                transform
                .original_size[1],
                transform
                .original_size[0],
            )
            if (
                mask.shape[:2]
                != expected_shape
            ):
                actual_size = (
                    int(
                        mask.shape[1]
                    ),
                    int(
                        mask.shape[0]
                    ),
                )
                raise ValueError(
                    "APDrawingGAN mask 尺寸 "
                    f"{actual_size} 与原图尺寸 "
                    f"{transform.original_size} 不一致"
                )

            left, top, right, bottom = (
                transform.crop_box
            )
            mask = mask[
                top:bottom,
                left:right,
            ]
            mask = cv2.resize(
                mask,
                transform.output_size,
                interpolation=(
                    cv2.INTER_NEAREST
                ),
            )

        if (
            mask.shape[:2]
            != (
                expected_size,
                expected_size,
            )
        ):
            raise ValueError(
                "APDrawingGAN mask 必须变换为 "
                f"{expected_size}x{expected_size}，"
                "实际为 "
                f"{mask.shape[1]}x{mask.shape[0]}"
            )

        write_image(
            destination,
            mask,
        )

    def _build_command(
        self,
        *,
        repo_wsl: str,
        input_wsl: str,
        landmark_wsl: str,
        mask_wsl: str,
        results_wsl: str,
    ) -> list[str]:
        return [
            self.config.wsl_executable,
            "-d",
            self.config.distro,
            "--cd",
            repo_wsl,
            "--",
            self.config.python_command,
            "test.py",
            "--dataroot",
            input_wsl,
            "--lm_dir",
            landmark_wsl,
            "--bg_dir",
            mask_wsl,
            "--name",
            self.config.experiment,
            "--model",
            "test",
            "--dataset_mode",
            "single",
            "--norm",
            "batch",
            "--use_local",
            "--which_epoch",
            self.config.epoch,
            "--gpu_ids",
            self.config.gpu_ids,
            "--how_many",
            "1",
            "--results_dir",
            results_wsl,
        ]

    def generate(
        self,
        request: GeneratorRequest,
    ) -> GeneratorOutput:
        (
            landmark_source,
            mask_source,
        ) = self._validate_request(
            request
        )
        points = self._load_landmarks(
            landmark_source
        )
        points = (
            self._transform_landmarks(
                points,
                request
                .preprocess_transform,
            )
        )

        if (
            np.any(points < 0)
            or np.any(
                points
                >= self.config.fine_size
            )
        ):
            raise ValueError(
                "变换后的 APDrawingGAN landmark "
                "超出 "
                f"{self.config.fine_size}x"
                f"{self.config.fine_size} "
                "输入范围"
            )

        work_dir = (
            Path(
                request.output_dir
            )
            / ".apdrawinggan_work"
        )
        shutil.rmtree(
            work_dir,
            ignore_errors=True,
        )

        input_dir = (
            work_dir
            / "input"
        )
        landmark_dir = (
            work_dir
            / "landmark"
        )
        mask_dir = (
            work_dir
            / "mask"
        )
        results_dir = (
            work_dir
            / "results"
        )

        staged_image = (
            input_dir
            / "portrait.png"
        )
        staged_landmark = (
            landmark_dir
            / "portrait.txt"
        )
        staged_mask = (
            mask_dir
            / "portrait.png"
        )

        write_image(
            staged_image,
            request.prepared_image,
        )
        self._write_landmarks(
            staged_landmark,
            points,
        )
        self._prepare_mask(
            mask_source,
            staged_mask,
            request
            .preprocess_transform,
            self.config.fine_size,
        )
        results_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        command = (
            self._build_command(
                repo_wsl=(
                    self._path_resolver(
                        self.config.repo
                    )
                ),
                input_wsl=(
                    self._path_resolver(
                        input_dir
                    )
                ),
                landmark_wsl=(
                    self._path_resolver(
                        landmark_dir
                    )
                ),
                mask_wsl=(
                    self._path_resolver(
                        mask_dir
                    )
                ),
                results_wsl=(
                    self._path_resolver(
                        results_dir
                    )
                ),
            )
        )

        try:
            completed = (
                self._runner(
                    command
                )
            )
        except (
            subprocess.TimeoutExpired
        ) as exc:
            raise RuntimeError(
                "APDrawingGAN 推理超时"
                f"（>{self.config.timeout_seconds:.0f}s），"
                "工作目录保留在："
                f"{work_dir}"
            ) from exc

        output_text = (
            self._combined_output(
                completed
            )
        )
        if (
            completed.returncode
            != 0
        ):
            output_tail = (
                output_text[-4000:]
                if output_text
                else "(无输出)"
            )
            raise RuntimeError(
                "APDrawingGAN 推理失败，"
                "returncode="
                f"{completed.returncode}。"
                "工作目录："
                f"{work_dir}\n"
                f"{output_tail}"
            )

        generated_path = (
            results_dir
            / self.config.experiment
            / (
                "test_"
                f"{self.config.epoch}"
            )
            / "images"
            / "portrait_fake_B.png"
        )
        if (
            not generated_path
            .is_file()
        ):
            raise RuntimeError(
                "APDrawingGAN 命令成功退出，"
                "但没有找到预期输出："
                f"{generated_path}。"
                "工作目录保留在："
                f"{work_dir}"
            )

        image = read_image(
            generated_path
        )
        metadata = {
            "backend":
                "apdrawinggan",
            "experiment":
                self.config.experiment,
            "epoch":
                self.config.epoch,
            "gpu_ids":
                self.config.gpu_ids,
            "distro":
                self.config.distro,
            "python":
                self.config
                .python_command,
            "input_size": [
                self.config.fine_size,
                self.config.fine_size,
            ],
            "auxiliary_inputs": {
                "landmark":
                    str(
                        landmark_source
                    ),
                "mask":
                    str(
                        mask_source
                    ),
            },
            "landmarks_transformed": (
                request
                .preprocess_transform
                is not None
            ),
            "seed":
                int(
                    request.seed
                ),
        }

        if (
            self.config
            .keep_work_dir
        ):
            metadata[
                "work_dir"
            ] = str(
                work_dir
            )
        else:
            shutil.rmtree(
                work_dir,
                ignore_errors=True,
            )

        return GeneratorOutput(
            image=image,
            metadata=metadata,
        )
