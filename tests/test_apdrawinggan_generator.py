from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest


ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(
        ROOT
        / "src"
    ),
)


from robot_portrait.generators import (
    APDrawingGANGenerator,
    APDrawingGANGeneratorConfig,
    GeneratorRequest,
    PreprocessTransform,
)
from robot_portrait.generators.apdrawinggan import (
    APDrawingGANPreflightReport,
    PreflightCheck,
)
from robot_portrait.io import (
    read_image,
    write_image,
)


class FakeRunner:
    def __init__(
        self,
        *,
        create_output: bool = True,
        returncode: int = 0,
        repo_visible: bool = True,
    ) -> None:
        self.commands: list[
            list[str]
        ] = []
        self.create_output = (
            create_output
        )
        self.returncode = (
            returncode
        )
        self.repo_visible = (
            repo_visible
        )

    def __call__(
        self,
        command,
    ) -> subprocess.CompletedProcess[str]:
        args = list(command)
        self.commands.append(
            args
        )

        if (
            "test" in args
            and "-f" in args
        ):
            return subprocess.CompletedProcess(
                args,
                (
                    0
                    if self.repo_visible
                    else 1
                ),
                "",
                "",
            )

        if "test.py" in args:
            if (
                self.create_output
                and self.returncode
                == 0
            ):
                results_dir = Path(
                    args[
                        args.index(
                            "--results_dir"
                        )
                        + 1
                    ]
                )
                name = args[
                    args.index(
                        "--name"
                    )
                    + 1
                ]
                epoch = args[
                    args.index(
                        "--which_epoch"
                    )
                    + 1
                ]

                output = (
                    results_dir
                    / name
                    / f"test_{epoch}"
                    / "images"
                    / "portrait_fake_B.png"
                )
                output.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                image = np.full(
                    (
                        512,
                        512,
                        3,
                    ),
                    255,
                    dtype=np.uint8,
                )
                image[
                    100:400,
                    220:292,
                ] = 0

                write_image(
                    output,
                    image,
                )

            return subprocess.CompletedProcess(
                args,
                self.returncode,
                (
                    "processing (0000)-th image"
                ),
                (
                    "boom"
                    if self.returncode
                    else ""
                ),
            )

        return subprocess.CompletedProcess(
            args,
            0,
            "",
            "",
        )


def _ready_preflight(
) -> APDrawingGANPreflightReport:
    return APDrawingGANPreflightReport(
        ready=True,
        checks=(),
    )


def _failed_preflight(
) -> APDrawingGANPreflightReport:
    return APDrawingGANPreflightReport(
        ready=False,
        checks=(
            PreflightCheck(
                name="torch_cuda",
                ok=False,
                required=True,
                message=(
                    "cuda unavailable"
                ),
            ),
        ),
    )


def _generator(
    tmp_path: Path,
    *,
    runner: FakeRunner | None = None,
    keep_work_dir: bool = False,
    preflight_provider=(
        _ready_preflight
    ),
):
    repo = (
        tmp_path
        / "APD 仓库"
    )
    repo.mkdir(
        exist_ok=True
    )
    (
        repo
        / "test.py"
    ).write_text(
        "# synthetic\n",
        encoding="utf-8",
    )

    runner = (
        runner
        or FakeRunner()
    )

    generator = (
        APDrawingGANGenerator(
            APDrawingGANGeneratorConfig(
                repo=repo,
                python_command=(
                    "/venv/bin/python"
                ),
                keep_work_dir=(
                    keep_work_dir
                ),
            ),
            runner=runner,
            path_resolver=(
                lambda path:
                    str(Path(path))
            ),
            preflight_provider=(
                preflight_provider
            ),
        )
    )

    return (
        generator,
        runner,
    )


def _auxiliary_files(
    tmp_path: Path,
    *,
    size: tuple[
        int,
        int,
    ] = (
        512,
        512,
    ),
) -> tuple[
    Path,
    Path,
]:
    landmark = (
        tmp_path
        / "五点.txt"
    )
    landmark.write_text(
        (
            "170 190\n"
            "342 190\n"
            "256 260\n"
            "220 340\n"
            "292 340\n"
        ),
        encoding="utf-8",
    )

    mask = (
        tmp_path
        / "背景.png"
    )
    image = np.full(
        (
            size[1],
            size[0],
        ),
        255,
        dtype=np.uint8,
    )
    image[
        :50,
        :,
    ] = 0
    write_image(
        mask,
        image,
    )

    return (
        landmark,
        mask,
    )


def _request(
    tmp_path: Path,
    landmark: Path,
    mask: Path,
    *,
    transform: (
        PreprocessTransform
        | None
    ) = None,
    image: np.ndarray | None = None,
) -> GeneratorRequest:
    if image is None:
        image = np.full(
            (
                512,
                512,
                3,
            ),
            240,
            dtype=np.uint8,
        )

    return GeneratorRequest(
        prepared_image=image,
        input_path=(
            tmp_path
            / "输入 人像.jpg"
        ),
        prepared_path=(
            tmp_path
            / "prepared.png"
        ),
        output_dir=(
            tmp_path
            / "输出 目录"
        ),
        seed=42,
        preprocess_transform=(
            transform
        ),
        auxiliary_inputs={
            "landmark":
                landmark,
            "mask":
                mask,
        },
    )


def test_status_ready(
    tmp_path,
):
    generator, _ = _generator(
        tmp_path
    )

    status = generator.status()

    assert (
        status.available
        is True
    )
    assert (
        status.details[
            "backend"
        ]
        == "apdrawinggan"
    )


def test_status_reports_preflight_failure(
    tmp_path,
):
    generator, _ = _generator(
        tmp_path,
        preflight_provider=(
            _failed_preflight
        ),
    )

    status = generator.status()

    assert (
        status.available
        is False
    )
    assert (
        "cuda unavailable"
        in (
            status.reason
            or ""
        )
    )


def test_status_reports_missing_wsl_mount(
    tmp_path,
):
    generator, _ = _generator(
        tmp_path,
        runner=FakeRunner(
            repo_visible=False
        ),
    )

    status = generator.status()

    assert (
        status.available
        is False
    )
    assert (
        "已挂载"
        in (
            status.reason
            or ""
        )
    )


def test_generate_success_with_unicode_paths_and_cleanup(
    tmp_path,
):
    landmark, mask = (
        _auxiliary_files(
            tmp_path
        )
    )
    generator, runner = (
        _generator(
            tmp_path
        )
    )

    output = generator.generate(
        _request(
            tmp_path,
            landmark,
            mask,
        )
    )

    assert (
        output.image.shape
        == (
            512,
            512,
            3,
        )
    )
    assert (
        output.metadata[
            "backend"
        ]
        == "apdrawinggan"
    )
    assert not (
        tmp_path
        / "输出 目录"
        / ".apdrawinggan_work"
    ).exists()

    command = next(
        command
        for command
        in runner.commands
        if "test.py"
        in command
    )
    assert (
        command[
            command.index(
                "--how_many"
            )
            + 1
        ]
        == "1"
    )
    assert (
        command[
            command.index(
                "--gpu_ids"
            )
            + 1
        ]
        == "0"
    )


def test_geometry_transforms_landmarks_and_mask(
    tmp_path,
):
    landmark = (
        tmp_path
        / "五点.txt"
    )
    landmark.write_text(
        (
            "300 200\n"
            "500 200\n"
            "400 300\n"
            "350 400\n"
            "450 400\n"
        ),
        encoding="utf-8",
    )

    mask = (
        tmp_path
        / "背景.png"
    )
    write_image(
        mask,
        np.full(
            (
                800,
                1000,
            ),
            255,
            dtype=np.uint8,
        ),
    )

    transform = (
        PreprocessTransform(
            original_size=(
                1000,
                800,
            ),
            output_size=(
                512,
                512,
            ),
            crop_box=(
                200,
                100,
                800,
                700,
            ),
        )
    )
    generator, _ = _generator(
        tmp_path,
        keep_work_dir=True,
    )

    generator.generate(
        _request(
            tmp_path,
            landmark,
            mask,
            transform=transform,
        )
    )

    work_dir = (
        tmp_path
        / "输出 目录"
        / ".apdrawinggan_work"
    )
    points = np.loadtxt(
        work_dir
        / "landmark"
        / "portrait.txt",
        dtype=int,
    )

    assert points.tolist() == [
        [85, 85],
        [256, 85],
        [171, 171],
        [128, 256],
        [213, 256],
    ]

    staged_mask = read_image(
        work_dir
        / "mask"
        / "portrait.png",
        cv2.IMREAD_GRAYSCALE,
    )
    assert (
        staged_mask.shape
        == (
            512,
            512,
        )
    )


def test_missing_auxiliary_inputs_are_actionable(
    tmp_path,
):
    generator, _ = _generator(
        tmp_path
    )
    request = GeneratorRequest(
        prepared_image=np.zeros(
            (
                512,
                512,
                3,
            ),
            dtype=np.uint8,
        ),
        input_path=(
            tmp_path
            / "input.png"
        ),
        prepared_path=(
            tmp_path
            / "prepared.png"
        ),
        output_dir=(
            tmp_path
            / "out"
        ),
        seed=1,
    )

    with pytest.raises(
        ValueError,
        match="landmark 和 mask",
    ):
        generator.generate(
            request
        )


def test_rejects_non_512_prepared_image(
    tmp_path,
):
    landmark, mask = (
        _auxiliary_files(
            tmp_path
        )
    )
    generator, _ = _generator(
        tmp_path
    )

    with pytest.raises(
        ValueError,
        match="512x512",
    ):
        generator.generate(
            _request(
                tmp_path,
                landmark,
                mask,
                image=np.zeros(
                    (
                        256,
                        256,
                        3,
                    ),
                    dtype=np.uint8,
                ),
            )
        )


def test_rejects_malformed_landmarks(
    tmp_path,
):
    landmark, mask = (
        _auxiliary_files(
            tmp_path
        )
    )
    landmark.write_text(
        "1 2\n3 4\n",
        encoding="utf-8",
    )
    generator, _ = _generator(
        tmp_path
    )

    with pytest.raises(
        ValueError,
        match="恰好包含 5",
    ):
        generator.generate(
            _request(
                tmp_path,
                landmark,
                mask,
            )
        )


def test_mask_must_match_original_size_before_transform(
    tmp_path,
):
    landmark, mask = (
        _auxiliary_files(
            tmp_path
        )
    )
    transform = (
        PreprocessTransform(
            original_size=(
                1000,
                800,
            ),
            output_size=(
                512,
                512,
            ),
            crop_box=(
                0,
                0,
                800,
                800,
            ),
        )
    )
    generator, _ = _generator(
        tmp_path
    )

    with pytest.raises(
        ValueError,
        match="与原图尺寸",
    ):
        generator.generate(
            _request(
                tmp_path,
                landmark,
                mask,
                transform=transform,
            )
        )


def test_subprocess_failure_preserves_work_dir(
    tmp_path,
):
    landmark, mask = (
        _auxiliary_files(
            tmp_path
        )
    )
    generator, _ = _generator(
        tmp_path,
        runner=FakeRunner(
            returncode=7
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="returncode=7",
    ):
        generator.generate(
            _request(
                tmp_path,
                landmark,
                mask,
            )
        )

    assert (
        tmp_path
        / "输出 目录"
        / ".apdrawinggan_work"
    ).exists()


def test_missing_expected_output_is_detected(
    tmp_path,
):
    landmark, mask = (
        _auxiliary_files(
            tmp_path
        )
    )
    generator, _ = _generator(
        tmp_path,
        runner=FakeRunner(
            create_output=False
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="没有找到预期输出",
    ):
        generator.generate(
            _request(
                tmp_path,
                landmark,
                mask,
            )
        )
def test_windows_drive_path_translation_preserves_unicode_and_spaces():
    translated = (
        APDrawingGANGenerator
        ._translate_windows_drive_path(
            r"H:\MySpace\Winter\智能机器人与智能信息系统\RobotPortraitArt"
        )
    )

    assert translated == (
        "/mnt/h/MySpace/Winter/"
        "智能机器人与智能信息系统/"
        "RobotPortraitArt"
    )


def test_windows_drive_path_translation_handles_root_and_posix_fallback():
    assert (
        APDrawingGANGenerator
        ._translate_windows_drive_path(
            r"C:\\"
        )
        == "/mnt/c"
    )

    assert (
        APDrawingGANGenerator
        ._translate_windows_drive_path(
            "/home/winter/project"
        )
        is None
    )
