"""Windows adapter for the official GANs N' Roses model running in WSL."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import cv2

from .apdrawinggan_backend import APDrawingGANGenerator
from .base import GeneratorOutput, GeneratorRequest, GeneratorStatus, GeneratorUnavailableError, PortraitGenerator
from ..preprocessing import load_portrait_image


class CartoonGenerator(PortraitGenerator):
    name = "cartoon"

    def __init__(self, root: Path, distro: str, python_command: str):
        self.root = Path(root)
        self.repo = self.root / "third_party/GANsNRoses"
        self.distro = distro
        self.python_command = python_command

    @staticmethod
    def _wsl_path(path: Path) -> str:
        resolved = path.resolve()
        translated = APDrawingGANGenerator._translate_windows_drive_path(resolved)
        if translated is None:
            raise ValueError(f"WSL model requires a Windows drive path: {resolved}")
        return translated

    def _run(self, *arguments: str):
        return subprocess.run([
            "wsl.exe", "-d", self.distro, "--exec", self.python_command,
            self._wsl_path(self.root / "scripts/run_cartoon_wsl.py"),
            "--repo", self._wsl_path(self.repo), *arguments,
        ], capture_output=True, timeout=180)

    @staticmethod
    def _error(result) -> str:
        # WSL diagnostics are UTF-16; Python tracebacks are UTF-8.
        data = result.stderr or result.stdout
        if b"Traceback" in data:
            data = data[data.index(b"Traceback"):]
        return data.decode("utf-16-le" if b"\x00" in data else "utf-8", errors="replace")[-3000:]

    def status(self) -> GeneratorStatus:
        details = {"backend": "gans-n-roses", "execution": "wsl-cuda", "input_size": 256,
                   "distro": self.distro, "python": self.python_command}
        required = ("model.py", "op/__init__.py", "op/fused_act.py", "op/upfirdn2d.py", "GNR_checkpoint.pt")
        missing = [name for name in required if not (self.repo / name).is_file()]
        if missing:
            return GeneratorStatus(False, "卡通模型未安装，请运行 .\\scripts\\start_web.ps1 -SetupCartoon", {**details, "missing": missing})
        try:
            result = self._run("--check")
            if result.returncode:
                raise RuntimeError(self._error(result))
            report = json.loads(result.stdout.decode("utf-8").splitlines()[-1])
            if report.get("ready") is not True:
                raise RuntimeError("Model check did not report ready")
            return GeneratorStatus(True, details={**details, **report})
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            return GeneratorStatus(False, f"卡通模型 WSL 检查失败：{exc}", details)

    def generate(self, request: GeneratorRequest) -> GeneratorOutput:
        if request.prepared_image.shape[:2] != (256, 256):
            raise ValueError("GANs N' Roses 卡通模型要求 size=256")
        target = request.output_dir / "cartoon.png"
        result = self._run("--input", self._wsl_path(request.prepared_path),
                           "--output", self._wsl_path(target), "--seed", str(request.seed))
        if result.returncode:
            raise GeneratorUnavailableError(self._error(result))
        image = load_portrait_image(target)
        if image.shape != request.prepared_image.shape:
            raise RuntimeError("Cartoon model returned an unexpected image shape")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Plotter paths follow cartoon contours, not filled areas of dark colour.
        lines = 255 - cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150, L2gradient=True)
        return GeneratorOutput(image=image, vector_image=lines, metadata={
            "backend": "gans-n-roses", "execution": "wsl-cuda", "seed": request.seed,
            "model_size": 256, "vector_source": "cartoon-contours-canny-50-150",
        }, warnings=("GANs N' Roses 的训练数据主要为女性自拍；其他人像和侧脸效果可能较差。",))
