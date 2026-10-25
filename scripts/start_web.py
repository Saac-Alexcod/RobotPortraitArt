"""Portable startup orchestration for the PowerShell entry point."""
from __future__ import annotations

import argparse
import errno
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".cache/web-runtime.json"
MARKER = "ROBOT_PORTRAIT_ENV="


def ensure_dependencies(check_only):
    if sys.version_info < (3, 10):
        raise RuntimeError("Windows Python 3.10+ is required. Recreate .venv with a supported Python.")
    required = ("uvicorn", "fastapi", "multipart", "numpy", "cv2", "PIL", "skimage", "yaml")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        if check_only:
            raise RuntimeError(f"Missing Windows packages: {', '.join(missing)}. Run start_web.ps1 to install.")
        print("[RobotPortraitArt] Installing Windows backend dependencies...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-r",
                        str(ROOT / "requirements-web.txt")], check=True)


def decode_wsl(data):
    return data.decode("utf-16-le" if b"\x00" in data else "utf-8", errors="replace").lstrip("\ufeff")


def resolve_distro(requested):
    if not shutil.which("wsl.exe"):
        raise RuntimeError("WSL is not installed. Baseline can run without WSL.")
    result = subprocess.run(["wsl.exe", "--list", "--quiet"], capture_output=True, timeout=30)
    if result.returncode:
        raise RuntimeError(decode_wsl(result.stderr or result.stdout).strip())
    names = [name.strip() for name in decode_wsl(result.stdout).splitlines() if name.strip()]
    if requested:
        if requested not in names:
            raise RuntimeError(f"WSL distro {requested!r} is not installed. Available: {', '.join(names)}")
        return requested
    for preferred in ("Ubuntu-22.04", "Ubuntu-24.04", "Ubuntu"):
        if preferred in names:
            return preferred
    usable = [name for name in names if not name.lower().startswith("docker-desktop")]
    if not usable:
        raise RuntimeError("No usable WSL distro. Install Ubuntu to enable APDrawingGAN.")
    return usable[0]


def inspect_apd(distro, python, setup):
    command = ["wsl.exe", "-d", distro, "--exec", "python3", "-"]
    if python:
        command += ["--python", python]
    if setup:
        command += ["--setup"]
        print("[RobotPortraitArt] Installing WSL APD environment (large CUDA download)...", flush=True)
    source = (ROOT / "scripts/apd_wsl_environment.py").read_bytes()
    if not setup:
        result = subprocess.run(command, input=source, capture_output=True, timeout=180)
        if result.stderr:
            print(decode_wsl(result.stderr).strip(), flush=True)
        for line in reversed(result.stdout.decode("utf-8", errors="replace").splitlines()):
            if line.startswith(MARKER):
                return json.loads(line[len(MARKER):])
        raise RuntimeError(f"WSL environment check exited with code {result.returncode}. "
                           "Ensure this distro has python3 installed.")
    # WSL diagnostics use UTF-16; Linux stdout uses UTF-8. Keep them separate.
    # Installation progress streams to the console instead of being buffered.
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=None)
    report = None
    try:
        process.stdin.write(source)
        process.stdin.close()
        for raw in process.stdout:
            line = raw.decode("utf-8", errors="replace").strip()
            if line.startswith(MARKER):
                report = json.loads(line[len(MARKER):])
            elif line:
                print(line, flush=True)
        code = process.wait()
    except BaseException:
        process.terminate()
        process.wait()
        raise
    finally:
        process.stdout.close()
    if report is None:
        raise RuntimeError(f"WSL environment check exited with code {code}. "
                           "Ensure this distro has python3 installed.")
    return report


def reserve_socket(host, port):
    """Keep the selected port bound until Uvicorn takes ownership."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    for candidate in range(port, min(port + 100, 65536)):
        sock = socket.socket(family)
        try:
            if os.name == "nt":
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            sock.bind((host, candidate))
            sock.set_inheritable(True)
            return sock
        except OSError as exc:
            sock.close()
            if exc.errno not in (errno.EADDRINUSE, errno.EACCES) and getattr(exc, "winerror", None) not in (10048, 10013):
                raise
    raise RuntimeError(f"No usable TCP port in {port}..{min(port + 99, 65535)}.")


def api_url(host, port):
    host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
    return f"http://{'[' + host + ']' if ':' in host else host}:{port}"


def publish_state(url, path=STATE):
    state = {"url": url, "pid": os.getpid(), "token": uuid.uuid4().hex}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{state['token']}.tmp")
    temporary.write_text(json.dumps(state), encoding="utf-8")
    temporary.replace(path)
    return state


def remove_state(state, path=STATE):
    try:
        if json.loads(path.read_text(encoding="utf-8")).get("token") == state["token"]:
            path.unlink()
    except (FileNotFoundError, ValueError):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--distro", default="")
    parser.add_argument("--apd-python", default="")
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--require-apdrawinggan", action="store_true")
    parser.add_argument("--setup-apd", action="store_true")
    parser.add_argument("--setup-cartoon", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.setup_cartoon and args.check_only:
        parser.error("-SetupCartoon cannot be combined with -CheckOnly")
    if not 1 <= args.port <= 65535:
        parser.error("port must be in 1..65535")
    if args.setup_apd and (args.check_only or args.apd_python):
        parser.error("-SetupAPD cannot be combined with -CheckOnly or an explicit APDPython/environment variable.")
    ensure_dependencies(args.check_only)
    print(f"[RobotPortraitArt] Repo: {ROOT}\nWindows Python: {sys.executable}", flush=True)
    try:
        distro = resolve_distro(args.distro)
        os.environ["ROBOT_PORTRAIT_APD_DISTRO"] = distro
        print(f"WSL distro: {distro}", flush=True)
        report = inspect_apd(distro, args.apd_python, args.setup_apd)
        os.environ["ROBOT_PORTRAIT_APD_PYTHON"] = report["python"]
        print(f"APD Python: {report['python']}", flush=True)
        if not report["ready"]:
            raise RuntimeError(report.get("error", "WSL Python/CUDA unavailable"))
        print(f"APD environment ready: torch {report['torch']}, {report['gpu']}", flush=True)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"WARNING: APDrawingGAN: {exc}", flush=True)
        print("To create its WSL environment: .\\scripts\\start_web.ps1 -SetupAPD\n"
              "Model weights are also required; see docs/MODEL_SETUP.md.", flush=True)
        if args.require_apdrawinggan or args.setup_apd or args.setup_cartoon:
            return 1
        print("Baseline remains available. /api/methods reports full model status.", flush=True)
    sys.path.insert(0, str(ROOT / "src"))
    if args.setup_cartoon:
        from setup_cartoon import setup
        setup()
        from robot_portrait.generators.cartoon import CartoonGenerator
        cartoon_status = CartoonGenerator(ROOT, distro, report["python"]).status()
        if not cartoon_status.available:
            print(f"Cartoon setup failed: {cartoon_status.reason}", flush=True)
            return 1
        print("GANs N' Roses WSL/CUDA model ready.", flush=True)
    if args.require_apdrawinggan:
        from robot_portrait.web.runtime import default_generator_statuses
        from robot_portrait.web.app import _effective_method_metadata
        status = _effective_method_metadata("apdrawinggan", default_generator_statuses()["apdrawinggan"])
        if not status["available"]:
            print(f"APDrawingGAN is required but unavailable: {status}", flush=True)
            return 1
    with reserve_socket(args.host, args.port) as sock:
        port = sock.getsockname()[1]
        url = api_url(args.host, port)
        if port != args.port:
            print(f"Port {args.port} is unavailable; selected {port}. Existing processes are left running.", flush=True)
        print(f"API: {url}\nDocs: {url}/docs", flush=True)
        if args.check_only:
            print("Check complete; no server started or packages installed.", flush=True)
            return 0
        import uvicorn
        config = uvicorn.Config("robot_portrait.web.app:app", host=args.host, port=port,
                                reload=args.reload, reload_dirs=[str(ROOT / "src")] if args.reload else None)
        server = uvicorn.Server(config)
        state = publish_state(url)
        try:
            if args.reload:
                from uvicorn.supervisors import ChangeReload
                ChangeReload(config, target=server.run, sockets=[sock]).run()
            else:
                server.run(sockets=[sock])
                if not server.started:
                    return 1
        finally:
            remove_state(state)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
