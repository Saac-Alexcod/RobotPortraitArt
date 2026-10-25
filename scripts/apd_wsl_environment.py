"""Executed via WSL stdin; requires only the Linux Python standard library."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

MARKER = "ROBOT_PORTRAIT_ENV="
PROBE = '''
import json, sys
result = {"python": sys.executable, "ready": False}
try:
    import torch, torchvision, dominate, numpy, PIL, cv2
    result.update(torch=torch.__version__, cuda=torch.cuda.is_available())
    if result["cuda"]:
        value = torch.ones(1, device="cuda") + 1
        torch.cuda.synchronize()
        result.update(ready=True, gpu=torch.cuda.get_device_name(0))
    else:
        result["error"] = "PyTorch CUDA is unavailable."
except Exception as exc:
    result["error"] = str(exc)
print("ROBOT_PORTRAIT_ENV=" + json.dumps(result))
'''


def probe(python):
    try:
        result = subprocess.run([str(python), "-c", PROBE], capture_output=True,
                                text=True, timeout=45)
        for line in reversed(result.stdout.splitlines()):
            if line.startswith(MARKER):
                return json.loads(line[len(MARKER):])
        return {"python": str(python), "ready": False,
                "error": (result.stderr or result.stdout or "Python probe failed")[-1500:]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"python": str(python), "ready": False, "error": str(exc)}


def candidates(home):
    paths = [home / ".venvs/robotportrait-apd/bin/python"]
    for variable in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        if os.environ.get(variable):
            paths.append(Path(os.environ[variable]) / "bin/python")
    registry = home / ".conda/environments.txt"
    if registry.is_file():
        paths.extend(Path(p) / "bin/python" for p in registry.read_text().splitlines() if p.strip())
    for pattern in (".venvs/*/bin/python", ".virtualenvs/*/bin/python",
                    ".conda/envs/*/bin/python", "miniconda3/envs/*/bin/python",
                    "anaconda3/envs/*/bin/python", "miniforge3/envs/*/bin/python",
                    "miniconda3/bin/python", "anaconda3/bin/python", "miniforge3/bin/python"):
        paths.extend(sorted(home.glob(pattern)))
    paths.append(Path(sys.executable))
    # Resolving symlinks would lose a venv's site-packages.
    return list(dict.fromkeys(str(p) for p in paths if p.is_file()))


def prepare_venv(directory):
    """Resume interrupted venv creation without clearing installed packages."""
    python = directory / "bin/python"
    # A previous venv attempt can leave bin/python behind even without pip.
    if python.exists():
        try:
            result = subprocess.run([str(python), "-m", "pip", "--version"],
                                    capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                return python
        except OSError:
            pass
    version = f"{sys.version_info[0]}.{sys.version_info[1]}"
    repair = (f"sudo apt-get update && sudo apt-get install -y python{version}-venv")
    if importlib.util.find_spec("venv") is None or importlib.util.find_spec("ensurepip") is None:
        raise RuntimeError(
            f"WSL Python {version} is missing venv/ensurepip. "
            f"In this WSL distro run: {repair}. "
            "Then rerun -SetupAPD; the incomplete environment will be repaired automatically."
        )
    # Running venv over an existing directory fills in missing pip files.
    # Do not use --clear: an existing environment may contain useful packages.
    result = subprocess.run([sys.executable, "-m", "venv", str(directory)])
    if result.returncode:
        raise RuntimeError(f"Cannot create/repair WSL venv at {directory}. "
                           f"On Ubuntu run: {repair}; then rerun -SetupAPD.")
    result = subprocess.run([str(python), "-m", "pip", "--version"],
                            capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError(f"WSL venv pip is still unavailable: {result.stderr.strip()}. "
                           f"On Ubuntu run: {repair}; then rerun -SetupAPD.")
    return python


def setup(home):
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        raise RuntimeError("Setup requires WSL Python 3.10-3.12 (e.g. Ubuntu 22.04/24.04). "
                           "Otherwise prepare an environment and pass -APDPython.")
    gpu = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    subprocess.run([gpu], check=True, timeout=30)
    directory = home / ".venvs/robotportrait-apd"
    python = prepare_venv(directory)
    commands = [
        ["-m", "pip", "install", "--upgrade", "pip"],
        ["-m", "pip", "install", "torch==2.5.1", "torchvision==0.20.1",
         "--index-url", "https://download.pytorch.org/whl/cu121"],
        ["-m", "pip", "install", "numpy<2", "Pillow>=10,<12", "dominate>=2.9,<3",
         "opencv-python-headless==4.10.0.84"],
    ]
    for command in commands:
        subprocess.run([str(python), *command], check=True)
    return probe(python)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="")
    parser.add_argument("--setup", action="store_true")
    args = parser.parse_args()
    home = Path.home()
    try:
        if args.setup:
            if args.python:
                raise RuntimeError("Use -SetupAPD without -APDPython; setup owns only ~/.venvs/robotportrait-apd.")
            result = setup(home)
        else:
            paths = [os.path.expanduser(args.python)] if args.python else candidates(home)
            reports = []
            for path in paths:
                report = probe(path)
                reports.append(report)
                if report["ready"]:
                    break
            result = dict(next((r for r in reports if r["ready"]), reports[0]))
            result["checked"] = reports
        print(MARKER + json.dumps(result), flush=True)
        return 0 if result["ready"] else 1
    except Exception as exc:
        attempted_python = str(home / ".venvs/robotportrait-apd/bin/python") if args.setup else (args.python or sys.executable)
        print(MARKER + json.dumps({"ready": False, "python": attempted_python, "error": str(exc)}), flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
