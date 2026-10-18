from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence


ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT / "src"),
)


from robot_portrait.generators.apdrawinggan import (
    APDrawingGANPreflight,
    APDrawingGANPreflightConfig,
)


def _repo(
    tmp_path: Path,
    *,
    with_checkpoints: bool = True,
) -> Path:
    repo = (
        tmp_path
        / "APDrawingGAN"
    )

    for relative in (
        "test.py",
        "options/base_options.py",
        "data/single_dataset.py",
        "models/test_model.py",
    ):
        path = (
            repo
            / relative
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            "# synthetic\n",
            encoding="utf-8",
        )

    if with_checkpoints:
        config = (
            APDrawingGANPreflightConfig(
                repo=repo
            )
        )

        for path in (
            config
            .expected_checkpoint_files()
        ):
            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            path.write_bytes(
                b"weights"
            )

    return repo


class FakeRunner:
    def __init__(
        self,
        *,
        modules: dict[
            str,
            bool,
        ] | None = None,
        cuda_available: bool = True,
        distro_ok: bool = True,
    ) -> None:
        self.modules = (
            modules
            or {
                "torch": True,
                "torchvision": True,
                "dominate": True,
                "numpy": True,
                "PIL": True,
            }
        )

        self.cuda_available = (
            cuda_available
        )

        self.distro_ok = (
            distro_ok
        )

        self.commands: list[
            list[str]
        ] = []

    def __call__(
        self,
        command: Sequence[str],
    ) -> subprocess.CompletedProcess[str]:
        args = list(command)

        self.commands.append(
            args
        )

        if args[-3:] == [
            "sh",
            "-lc",
            "printf ready",
        ]:
            return (
                subprocess.CompletedProcess(
                    args,
                    (
                        0
                        if self.distro_ok
                        else 1
                    ),
                    (
                        "ready"
                        if self.distro_ok
                        else ""
                    ),
                    (
                        ""
                        if self.distro_ok
                        else "missing distro"
                    ),
                )
            )

        if (
            "--query-gpu=name,memory.total"
            in args
        ):
            return (
                subprocess.CompletedProcess(
                    args,
                    (
                        0
                        if self.cuda_available
                        else 1
                    ),
                    (
                        "NVIDIA GeForce RTX 4060 "
                        "Laptop GPU, 8188 MiB\n"
                        if self.cuda_available
                        else ""
                    ),
                    (
                        ""
                        if self.cuda_available
                        else "gpu unavailable"
                    ),
                )
            )

        code = (
            args[-1]
            if (
                args
                and args[-2:-1]
                == ["-c"]
            )
            else ""
        )

        if (
            "importlib.util"
            in code
        ):
            return (
                subprocess.CompletedProcess(
                    args,
                    0,
                    json.dumps(
                        self.modules
                    ),
                    "",
                )
            )

        if (
            "import json,torch"
            in code
        ):
            if not self.modules.get(
                "torch",
                False,
            ):
                return (
                    subprocess.CompletedProcess(
                        args,
                        1,
                        "",
                        (
                            "ModuleNotFoundError: "
                            "No module named 'torch'"
                        ),
                    )
                )

            return (
                subprocess.CompletedProcess(
                    args,
                    0,
                    json.dumps(
                        {
                            "torch":
                                "2.9.0",

                            "cuda_available":
                                self.cuda_available,

                            "torch_cuda":
                                "13.0",

                            "device_count":
                                (
                                    1
                                    if self.cuda_available
                                    else 0
                                ),

                            "device_name":
                                (
                                    "NVIDIA GeForce "
                                    "RTX 4060 Laptop GPU"
                                    if self.cuda_available
                                    else None
                                ),
                        }
                    ),
                    "",
                )
            )

        if (
            "sys.version.split"
            in code
        ):
            return (
                subprocess.CompletedProcess(
                    args,
                    0,
                    json.dumps(
                        {
                            "version":
                                "3.10.12",

                            "executable":
                                "/usr/bin/python3",
                        }
                    ),
                    "",
                )
            )

        raise AssertionError(
            f"unexpected command: {args}"
        )


def _preflight(
    repo: Path,
    runner: FakeRunner,
    *,
    require_cuda: bool = True,
) -> APDrawingGANPreflight:
    return APDrawingGANPreflight(
        APDrawingGANPreflightConfig(
            repo=repo,
            require_cuda=require_cuda,
        ),

        runner=runner,

        which=lambda _:
            r"C:\Windows\System32\wsl.exe",
    )


def test_expected_checkpoint_set_contains_all_local_generators(
    tmp_path,
):
    config = (
        APDrawingGANPreflightConfig(
            repo=tmp_path
        )
    )

    assert [
        path.name
        for path
        in config
        .expected_checkpoint_files()
    ] == [
        "300_net_G.pth",
        "300_net_GLEyel.pth",
        "300_net_GLEyer.pth",
        "300_net_GLNose.pth",
        "300_net_GLMouth.pth",
        "300_net_GLHair.pth",
        "300_net_GLBG.pth",
        "300_net_GCombine.pth",
    ]


def test_preflight_reports_missing_repo_files(
    tmp_path,
):
    report = _preflight(
        tmp_path / "missing",
        FakeRunner(),
    ).run()

    check = next(
        check
        for check in report.checks
        if check.name
        == "repo_layout"
    )

    assert check.ok is False

    assert (
        "test.py"
        in check.details["missing"]
    )

    assert report.ready is False


def test_preflight_reports_all_missing_checkpoints(
    tmp_path,
):
    repo = _repo(
        tmp_path,
        with_checkpoints=False,
    )

    report = _preflight(
        repo,
        FakeRunner(),
    ).run()

    check = next(
        check
        for check in report.checks
        if check.name
        == "checkpoints"
    )

    assert check.ok is False

    assert (
        check.details[
            "expected_count"
        ]
        == 8
    )

    assert (
        len(
            check.details[
                "missing"
            ]
        )
        == 8
    )

    assert report.ready is False


def test_preflight_stops_cleanly_when_wsl_is_missing(
    tmp_path,
):
    repo = _repo(
        tmp_path
    )

    report = APDrawingGANPreflight(
        APDrawingGANPreflightConfig(
            repo=repo
        ),

        runner=FakeRunner(),

        which=lambda _: None,
    ).run()

    assert [
        check.name
        for check
        in report.checks
    ] == [
        "repo_layout",
        "checkpoints",
        "wsl_executable",
    ]

    assert report.ready is False


def test_preflight_reports_missing_python_module(
    tmp_path,
):
    repo = _repo(
        tmp_path
    )

    modules = {
        "torch": False,
        "torchvision": False,
        "dominate": True,
        "numpy": True,
        "PIL": True,
    }

    report = _preflight(
        repo,
        FakeRunner(
            modules=modules
        ),
    ).run()

    check = next(
        check
        for check in report.checks
        if check.name
        == "python_modules"
    )

    assert check.ok is False

    assert (
        check.details[
            "missing"
        ]
        == [
            "torch",
            "torchvision",
        ]
    )

    assert report.ready is False


def test_preflight_requires_cuda_by_default(
    tmp_path,
):
    repo = _repo(
        tmp_path
    )

    report = _preflight(
        repo,
        FakeRunner(
            cuda_available=False
        ),
    ).run()

    check = next(
        check
        for check in report.checks
        if check.name
        == "torch_cuda"
    )

    assert check.ok is False
    assert report.ready is False


def test_preflight_can_allow_cpu_mode(
    tmp_path,
):
    repo = _repo(
        tmp_path
    )

    report = _preflight(
        repo,
        FakeRunner(
            cuda_available=False
        ),
        require_cuda=False,
    ).run()

    torch_check = next(
        check
        for check in report.checks
        if check.name
        == "torch_cuda"
    )

    gpu_check = next(
        check
        for check in report.checks
        if check.name
        == "nvidia_smi"
    )

    assert torch_check.ok is True

    assert (
        gpu_check.required
        is False
    )

    assert report.ready is True


def test_preflight_ready_with_repo_weights_wsl_modules_and_cuda(
    tmp_path,
):
    repo = _repo(
        tmp_path
    )

    report = _preflight(
        repo,
        FakeRunner(),
    ).run()

    assert report.ready is True

    assert (
        report
        .failed_required_checks()
        == ()
    )

    assert all(
        check.ok
        for check in report.checks
        if check.required
    )