from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from urllib.request import urlopen

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = load_script("start_web")
environment = load_script("apd_wsl_environment")


def test_busy_port_is_skipped_and_reserved():
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        port = occupied.getsockname()[1]
        if port == 65535:
            pytest.skip("OS selected the final port")
        with launcher.reserve_socket("127.0.0.1", port) as reserved:
            assert reserved.getsockname()[1] > port
            with socket.socket() as competitor, pytest.raises(OSError):
                competitor.bind(reserved.getsockname())
        # The launcher must never stop the process that owns the original port.
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            pass


def test_older_server_cannot_remove_newer_server_state(tmp_path):
    path = tmp_path / "runtime.json"
    first = launcher.publish_state("http://127.0.0.1:8000", path)
    second = launcher.publish_state("http://127.0.0.1:8001", path)
    launcher.remove_state(first, path)
    assert json.loads(path.read_text())["url"] == "http://127.0.0.1:8001"
    launcher.remove_state(second, path)
    assert not path.exists()


def test_distro_discovery_handles_utf16_and_explicit_override(monkeypatch):
    monkeypatch.setattr(launcher.shutil, "which", lambda _: "wsl.exe")
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(
        a, 0, "docker-desktop\r\nUbuntu-24.04\r\n".encode("utf-16-le"), b""))
    assert launcher.resolve_distro("") == "Ubuntu-24.04"
    assert launcher.resolve_distro("Ubuntu-24.04") == "Ubuntu-24.04"
    with pytest.raises(RuntimeError, match="not installed"):
        launcher.resolve_distro("Ubuntu-22.04")


def test_discovery_selects_cuda_environment_instead_of_system_python(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["environment"])
    monkeypatch.setattr(environment, "candidates", lambda _: ["/system/python", "/home/user/env/bin/python"])
    monkeypatch.setattr(environment, "probe", lambda p: {"python": p, "ready": p.startswith("/home/")})
    assert environment.main() == 0
    report = json.loads(capsys.readouterr().out.split(environment.MARKER)[1])
    assert report["python"] == "/home/user/env/bin/python"
    assert len(report["checked"]) == 2


def test_bad_explicit_python_is_not_silently_replaced(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["environment", "--python", "/missing/bin/python"])
    monkeypatch.setattr(environment, "candidates", lambda _: pytest.fail("explicit path must win"))
    monkeypatch.setattr(environment, "probe", lambda p: {"python": p, "ready": False, "error": "missing"})
    assert environment.main() == 1
    report = json.loads(capsys.readouterr().out.split(environment.MARKER)[1])
    assert report["python"] == "/missing/bin/python"


def test_setup_creates_only_managed_venv_and_installs_cuda_pair(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(sys, "version_info", (3, 10, 12))
    monkeypatch.setattr(environment.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(environment.shutil, "which", lambda _: "/usr/lib/wsl/lib/nvidia-smi")
    monkeypatch.setattr(environment.subprocess, "run", lambda cmd, **kw: (
        commands.append(cmd) or subprocess.CompletedProcess(cmd, 0)))
    monkeypatch.setattr(environment, "probe", lambda p: {"python": str(p), "ready": True})
    result = environment.setup(tmp_path)
    python = str(tmp_path / ".venvs/robotportrait-apd/bin/python")
    assert result["python"] == python
    installs = [cmd for cmd in commands if "pip" in cmd]
    assert all(cmd[0] == python for cmd in installs)
    assert any("torch==2.5.1" in cmd and "torchvision==0.20.1" in cmd for cmd in installs)


def test_failed_setup_does_not_continue_installing(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(sys, "version_info", (3, 10, 12))
    monkeypatch.setattr(environment.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(environment.shutil, "which", lambda _: "nvidia-smi")
    monkeypatch.setattr(environment.subprocess, "run", lambda cmd, **kw: (
        calls.append(cmd) or subprocess.CompletedProcess(cmd, 1 if "venv" in cmd else 0)))
    with pytest.raises(RuntimeError, match="python3.10-venv"):
        environment.setup(tmp_path)
    assert not any("pip" in cmd for cmd in calls)


def test_partial_venv_reports_missing_ensurepip_before_install(monkeypatch, tmp_path):
    python = tmp_path / "bin/python"
    python.parent.mkdir()
    python.touch()
    monkeypatch.setattr(sys, "version_info", (3, 10, 12))
    monkeypatch.setattr(environment.importlib.util, "find_spec", lambda name: None if name == "ensurepip" else object())
    calls = []
    def run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 1, "", "No module named pip")
    monkeypatch.setattr(environment.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="python3.10-venv"):
        environment.prepare_venv(tmp_path)
    assert calls == [[str(python), "-m", "pip", "--version"]]


def test_partial_venv_is_repaired_without_deleting_packages(monkeypatch, tmp_path):
    python = tmp_path / "bin/python"
    python.parent.mkdir()
    python.touch()
    sentinel = tmp_path / "existing-package.txt"
    sentinel.write_text("keep")
    monkeypatch.setattr(environment.importlib.util, "find_spec", lambda _: object())
    calls = []
    def run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 1 if len(calls) == 1 else 0, "", "")
    monkeypatch.setattr(environment.subprocess, "run", run)
    assert environment.prepare_venv(tmp_path) == python
    assert calls[1] == [sys.executable, "-m", "venv", str(tmp_path)]
    assert calls[2] == [str(python), "-m", "pip", "--version"]
    assert sentinel.read_text() == "keep"


def test_working_pip_does_not_require_ensurepip(monkeypatch, tmp_path):
    python = tmp_path / "bin/python"
    python.parent.mkdir()
    python.touch()
    monkeypatch.setattr(environment.importlib.util, "find_spec", lambda _: pytest.fail("pip works already"))
    monkeypatch.setattr(environment.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0))
    assert environment.prepare_venv(tmp_path) == python


def test_check_only_and_setup_are_mutually_exclusive(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["launcher", "--check-only", "--setup-apd"])
    monkeypatch.setattr(launcher, "ensure_dependencies", lambda _: pytest.fail("must not install"))
    with pytest.raises(SystemExit) as exc:
        launcher.main()
    assert exc.value.code == 2


def test_check_only_rejects_cartoon_download(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["launcher", "--check-only", "--setup-cartoon"])
    monkeypatch.setattr(launcher, "ensure_dependencies", lambda _: pytest.fail("must not install"))
    with pytest.raises(SystemExit) as exc:
        launcher.main()
    assert exc.value.code == 2


def test_strict_check_requires_web_preprocessing_models(monkeypatch):
    from robot_portrait.web import runtime
    monkeypatch.setattr(sys, "argv", ["launcher", "--check-only", "--require-apdrawinggan"])
    monkeypatch.setattr(launcher, "ensure_dependencies", lambda _: None)
    monkeypatch.setattr(launcher, "resolve_distro", lambda _: "Ubuntu")
    monkeypatch.setattr(launcher, "inspect_apd", lambda *args: {
        "python": "/env/bin/python", "ready": True, "torch": "test", "gpu": "test"})
    monkeypatch.setattr(runtime, "default_generator_statuses", lambda: {
        "apdrawinggan": {"available": True, "details": {"auto_preprocess": {"available": False}}}})
    monkeypatch.setattr(launcher, "reserve_socket", lambda *args: pytest.fail("must fail before serving"))
    assert launcher.main() == 1


def test_strict_environment_failure_does_not_start_server(monkeypatch):
    monkeypatch.setenv("ROBOT_PORTRAIT_APD_DISTRO", "")
    monkeypatch.setenv("ROBOT_PORTRAIT_APD_PYTHON", "")
    monkeypatch.setattr(sys, "argv", ["launcher", "--require-apdrawinggan"])
    monkeypatch.setattr(launcher, "ensure_dependencies", lambda _: None)
    monkeypatch.setattr(launcher, "resolve_distro", lambda _: "Ubuntu")
    monkeypatch.setattr(launcher, "inspect_apd", lambda *args: {
        "python": "/usr/bin/python3", "ready": False, "error": "No torch"})
    monkeypatch.setattr(launcher, "reserve_socket", lambda *args: pytest.fail("must not start"))
    assert launcher.main() == 1


def test_wildcard_and_ipv6_urls_are_connectable():
    assert launcher.api_url("0.0.0.0", 8000) == "http://127.0.0.1:8000"
    assert launcher.api_url("::", 8001) == "http://[::1]:8001"


def test_launcher_serves_health_on_fallback_port_and_cleans_state(monkeypatch, tmp_path):
    import uvicorn

    monkeypatch.setenv("ROBOT_PORTRAIT_APD_DISTRO", "")
    monkeypatch.setenv("ROBOT_PORTRAIT_APD_PYTHON", "")
    monkeypatch.setattr(launcher, "ensure_dependencies", lambda _: None)
    monkeypatch.setattr(launcher, "resolve_distro", lambda _: "Ubuntu")
    monkeypatch.setattr(launcher, "inspect_apd", lambda *args: {
        "python": "/env/bin/python", "ready": True, "torch": "test", "gpu": "test"})
    path = tmp_path / "runtime.json"
    publish, remove = launcher.publish_state, launcher.remove_state
    monkeypatch.setattr(launcher, "publish_state", lambda url: publish(url, path))
    monkeypatch.setattr(launcher, "remove_state", lambda state: remove(state, path))
    server_type = uvicorn.Server
    responses, errors, threads = [], [], []

    def create_server(config):
        server = server_type(config)

        def request_health():
            try:
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    try:
                        with urlopen(f"http://127.0.0.1:{config.port}/api/health", timeout=1) as response:
                            responses.append(json.load(response))
                        return
                    except OSError:
                        time.sleep(0.05)
                raise AssertionError("Server did not become healthy")
            except Exception as exc:
                errors.append(exc)
            finally:
                server.should_exit = True

        thread = threading.Thread(target=request_health, daemon=True)
        threads.append(thread)
        thread.start()
        return server

    monkeypatch.setattr(uvicorn, "Server", create_server)
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        port = occupied.getsockname()[1]
        if port == 65535:
            pytest.skip("OS selected final port")
        monkeypatch.setattr(sys, "argv", ["launcher", "--port", str(port)])
        assert launcher.main() == 0
    for thread in threads:
        thread.join(timeout=2)
    assert not errors
    assert responses == [{"status": "ok", "service": "robot-portrait-api"}]
    assert not path.exists()
