from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable, Sequence


CommandRunner = Callable[
    [Sequence[str]],
    subprocess.CompletedProcess[str],
]


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    """One APDrawingGAN environment check."""

    name: str
    ok: bool
    required: bool
    message: str

    details: dict[str, Any] = field(
        default_factory=dict
    )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "required": self.required,
            "message": self.message,
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class APDrawingGANPreflightReport:
    """JSON-ready APDrawingGAN environment report."""

    ready: bool
    checks: tuple[PreflightCheck, ...]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "checks": [
                check.to_metadata()
                for check in self.checks
            ],
        }

    def failed_required_checks(
        self,
    ) -> tuple[PreflightCheck, ...]:
        return tuple(
            check
            for check in self.checks
            if check.required and not check.ok
        )


@dataclass(frozen=True, slots=True)
class APDrawingGANPreflightConfig:
    """Static APDrawingGAN environment configuration."""

    repo: Path

    distro: str = "Ubuntu-22.04"
    python_command: str = "python3"

    experiment: str = "formal_author"
    epoch: str = "300"

    wsl_executable: str = "wsl.exe"

    require_cuda: bool = True

    def checkpoint_dir(self) -> Path:
        return (
            self.repo
            / "checkpoints"
            / self.experiment
        )

    def expected_checkpoint_files(
        self,
    ) -> tuple[Path, ...]:
        network_names = (
            "G",
            "GLEyel",
            "GLEyer",
            "GLNose",
            "GLMouth",
            "GLHair",
            "GLBG",
            "GCombine",
        )

        return tuple(
            self.checkpoint_dir()
            / f"{self.epoch}_net_{name}.pth"

            for name in network_names
        )


class APDrawingGANPreflight:
    """Inspect the legacy APDrawingGAN runtime.

    This diagnostic layer does not install or modify anything.
    """

    _REQUIRED_REPO_FILES = (
        "test.py",
        "options/base_options.py",
        "data/single_dataset.py",
        "models/test_model.py",
    )

    _PYTHON_MODULES = (
        "torch",
        "torchvision",
        "dominate",
        "numpy",
        "PIL",
    )

    def __init__(
        self,
        config: APDrawingGANPreflightConfig,
        *,
        runner: CommandRunner | None = None,
        which: Callable[
            [str],
            str | None,
        ] = shutil.which,
    ) -> None:
        self.config = config

        self._runner = (
            runner
            or self._default_runner
        )

        self._which = which

    @staticmethod
    def _default_runner(
        command: Sequence[str],
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )

    @staticmethod
    def _output(
        completed: subprocess.CompletedProcess[str],
    ) -> str:
        return (
            completed.stdout
            or completed.stderr
            or ""
        ).strip()

    def _run_wsl(
        self,
        *command: str,
    ) -> subprocess.CompletedProcess[str]:
        return self._runner(
            [
                self.config.wsl_executable,
                "-d",
                self.config.distro,
                "--",
                *command,
            ]
        )

    def _check_repo(
        self,
    ) -> PreflightCheck:
        repo = self.config.repo

        missing = [
            relative
            for relative
            in self._REQUIRED_REPO_FILES
            if not (
                repo / relative
            ).is_file()
        ]

        if missing:
            return PreflightCheck(
                name="repo_layout",
                ok=False,
                required=True,
                message=(
                    "APDrawingGAN 源码目录不完整。"
                ),
                details={
                    "repo": str(repo),
                    "missing": missing,
                },
            )

        return PreflightCheck(
            name="repo_layout",
            ok=True,
            required=True,
            message=(
                "APDrawingGAN 源码结构完整。"
            ),
            details={
                "repo": str(repo),
            },
        )

    def _check_checkpoints(
        self,
    ) -> PreflightCheck:
        expected = (
            self.config
            .expected_checkpoint_files()
        )

        missing = [
            path.name
            for path in expected
            if not path.is_file()
        ]

        if missing:
            return PreflightCheck(
                name="checkpoints",
                ok=False,
                required=True,
                message=(
                    "APDrawingGAN 预训练权重不完整，"
                    f"缺少 {len(missing)} 个文件。"
                ),
                details={
                    "directory": str(
                        self.config
                        .checkpoint_dir()
                    ),
                    "expected_count":
                        len(expected),
                    "missing":
                        missing,
                },
            )

        return PreflightCheck(
            name="checkpoints",
            ok=True,
            required=True,
            message=(
                "APDrawingGAN 预训练权重文件齐全。"
            ),
            details={
                "directory": str(
                    self.config
                    .checkpoint_dir()
                ),
                "expected_count":
                    len(expected),
            },
        )

    def _check_wsl_executable(
        self,
    ) -> PreflightCheck:
        resolved = self._which(
            self.config.wsl_executable
        )

        if not resolved:
            return PreflightCheck(
                name="wsl_executable",
                ok=False,
                required=True,
                message="未找到 wsl.exe。",
                details={
                    "executable":
                        self.config
                        .wsl_executable,
                },
            )

        return PreflightCheck(
            name="wsl_executable",
            ok=True,
            required=True,
            message="已找到 WSL 启动器。",
            details={
                "path": resolved,
            },
        )

    def _check_wsl_distro(
        self,
    ) -> PreflightCheck:
        completed = self._run_wsl(
            "sh",
            "-lc",
            "printf ready",
        )

        ok = (
            completed.returncode == 0
            and self._output(
                completed
            ) == "ready"
        )

        return PreflightCheck(
            name="wsl_distro",
            ok=ok,
            required=True,
            message=(
                f"WSL 发行版 "
                f"{self.config.distro} 可运行。"
                if ok
                else
                f"无法启动 WSL 发行版 "
                f"{self.config.distro}。"
            ),
            details={
                "distro":
                    self.config.distro,
                "returncode":
                    completed.returncode,
                "output":
                    self._output(completed),
            },
        )

    def _check_python(
        self,
    ) -> PreflightCheck:
        code = (
            "import json,sys; "
            "print(json.dumps({"
            "'version':sys.version.split()[0],"
            "'executable':sys.executable"
            "}))"
        )

        completed = self._run_wsl(
            self.config.python_command,
            "-c",
            code,
        )

        details: dict[
            str,
            Any,
        ] = {
            "command":
                self.config.python_command,

            "returncode":
                completed.returncode,
        }

        if completed.returncode == 0:
            try:
                details.update(
                    json.loads(
                        completed
                        .stdout
                        .strip()
                    )
                )

            except (
                json.JSONDecodeError,
                TypeError,
            ):
                details["output"] = (
                    self._output(
                        completed
                    )
                )

        return PreflightCheck(
            name="python",
            ok=(
                completed.returncode
                == 0
            ),
            required=True,
            message=(
                "WSL Python 可运行。"
                if completed.returncode == 0
                else "WSL Python 无法运行。"
            ),
            details=details,
        )

    def _check_python_modules(
        self,
    ) -> PreflightCheck:
        modules_json = json.dumps(
            self._PYTHON_MODULES
        )

        code = (
            "import importlib.util,json; "
            f"mods={modules_json}; "
            "print(json.dumps({"
            "m:(importlib.util.find_spec(m) "
            "is not None) for m in mods"
            "}))"
        )

        completed = self._run_wsl(
            self.config.python_command,
            "-c",
            code,
        )

        if completed.returncode != 0:
            return PreflightCheck(
                name="python_modules",
                ok=False,
                required=True,
                message=(
                    "无法检查 WSL Python 依赖。"
                ),
                details={
                    "returncode":
                        completed.returncode,
                    "output":
                        self._output(
                            completed
                        ),
                },
            )

        try:
            modules = json.loads(
                completed
                .stdout
                .strip()
            )

        except json.JSONDecodeError:
            return PreflightCheck(
                name="python_modules",
                ok=False,
                required=True,
                message=(
                    "WSL Python 依赖检查输出"
                    "无法解析。"
                ),
                details={
                    "output":
                        self._output(
                            completed
                        )
                },
            )

        missing = [
            module
            for module
            in self._PYTHON_MODULES
            if not modules.get(
                module,
                False,
            )
        ]

        return PreflightCheck(
            name="python_modules",
            ok=not missing,
            required=True,
            message=(
                "APDrawingGAN 推理所需 "
                "Python 模块已安装。"
                if not missing
                else
                "APDrawingGAN 推理所需 "
                "Python 模块尚未齐全。"
            ),
            details={
                "modules": modules,
                "missing": missing,
            },
        )

    def _check_torch_cuda(
        self,
    ) -> PreflightCheck:
        code = (
            "import json,torch; "
            "print(json.dumps({"
            "'torch':torch.__version__,"
            "'cuda_available':"
            "torch.cuda.is_available(),"
            "'torch_cuda':torch.version.cuda,"
            "'device_count':"
            "torch.cuda.device_count(),"
            "'device_name':("
            "torch.cuda.get_device_name(0) "
            "if torch.cuda.is_available() "
            "else None)"
            "}))"
        )

        completed = self._run_wsl(
            self.config.python_command,
            "-c",
            code,
        )

        if completed.returncode != 0:
            return PreflightCheck(
                name="torch_cuda",
                ok=False,
                required=True,
                message=(
                    "无法导入 torch 或查询 CUDA。"
                ),
                details={
                    "returncode":
                        completed.returncode,
                    "output":
                        self._output(
                            completed
                        ),
                },
            )

        try:
            details = json.loads(
                completed
                .stdout
                .strip()
            )

        except json.JSONDecodeError:
            return PreflightCheck(
                name="torch_cuda",
                ok=False,
                required=True,
                message=(
                    "torch/CUDA 检查输出"
                    "无法解析。"
                ),
                details={
                    "output":
                        self._output(
                            completed
                        )
                },
            )

        cuda_available = bool(
            details.get(
                "cuda_available"
            )
        )

        ok = (
            cuda_available
            or not self.config
            .require_cuda
        )

        return PreflightCheck(
            name="torch_cuda",
            ok=ok,
            required=True,
            message=(
                "PyTorch CUDA 可用."
                if cuda_available
                else (
                    "PyTorch 可运行，但 CUDA 不可用。"
                    if not self.config.require_cuda
                    else "PyTorch CUDA 不可用。"
                )
            ),
            details=details,
        )

    def _check_nvidia_smi(
        self,
    ) -> PreflightCheck:
        commands = (
            "/usr/lib/wsl/lib/nvidia-smi",
            "nvidia-smi",
        )

        last: (
            subprocess.CompletedProcess[str]
            | None
        ) = None

        for executable in commands:
            completed = self._run_wsl(
                executable,
                "--query-gpu=name,memory.total",
                "--format=csv,noheader",
            )

            last = completed

            if completed.returncode == 0:
                return PreflightCheck(
                    name="nvidia_smi",
                    ok=True,
                    required=(
                        self.config
                        .require_cuda
                    ),
                    message=(
                        "WSL 能访问 NVIDIA GPU。"
                    ),
                    details={
                        "command":
                            executable,
                        "gpu":
                            self._output(
                                completed
                            ),
                    },
                )

        assert last is not None

        return PreflightCheck(
            name="nvidia_smi",
            ok=False,
            required=(
                self.config
                .require_cuda
            ),
            message=(
                "WSL 无法通过 nvidia-smi "
                "访问 NVIDIA GPU。"
            ),
            details={
                "returncode":
                    last.returncode,
                "output":
                    self._output(
                        last
                    ),
            },
        )

    def run(
        self,
    ) -> APDrawingGANPreflightReport:
        checks: list[
            PreflightCheck
        ] = [
            self._check_repo(),
            self._check_checkpoints(),
            self._check_wsl_executable(),
        ]

        if not checks[-1].ok:
            return (
                APDrawingGANPreflightReport(
                    ready=False,
                    checks=tuple(checks),
                )
            )

        distro_check = (
            self._check_wsl_distro()
        )

        checks.append(
            distro_check
        )

        if not distro_check.ok:
            return (
                APDrawingGANPreflightReport(
                    ready=False,
                    checks=tuple(checks),
                )
            )

        python_check = (
            self._check_python()
        )

        checks.append(
            python_check
        )

        if not python_check.ok:
            return (
                APDrawingGANPreflightReport(
                    ready=False,
                    checks=tuple(checks),
                )
            )

        checks.append(
            self._check_python_modules()
        )

        checks.append(
            self._check_nvidia_smi()
        )

        checks.append(
            self._check_torch_cuda()
        )

        ready = all(
            check.ok
            for check in checks
            if check.required
        )

        return (
            APDrawingGANPreflightReport(
                ready=ready,
                checks=tuple(checks),
            )
        )