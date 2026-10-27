"""Installer command telemetry and failure tests; Windows, network, WSL not required."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "installer_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("rpa_installer_diagnostics", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_command_output_and_success(tmp_path: Path, capsys):
    log = tmp_path / "step.log"
    status = module.run_step("sample", [sys.executable, "-c", "print('HELLO_FROM_CHILD')"], log, 5)
    assert status == 0
    assert "HELLO_FROM_CHILD" in log.read_text()
    assert "END sample: exit=0" in capsys.readouterr().out


def test_failure_code_logged(tmp_path: Path):
    log = tmp_path / "failed.log"
    status = module.run_step("broken", [sys.executable, "-c", "raise SystemExit(17)"], log, 5)
    assert status == 17
    assert "exit=17" in log.read_text()


def test_hanging_process_times_out(tmp_path: Path):
    log = tmp_path / "timeout.log"
    status = module.run_step("stuck", [sys.executable, "-c", "import time;time.sleep(30)"], log, 1)
    assert status == 124
    assert "RPA-E120" in log.read_text()


def test_stdin_to_child(tmp_path: Path):
    input_file = tmp_path / "input.txt"
    input_file.write_text("my-data", encoding="utf-8")
    log = tmp_path / "stdin.log"
    status = module.run_step("stdin", [sys.executable, "-c", "import sys;print(sys.stdin.read())"], log, 5, input_file)
    assert status == 0
    assert "my-data" in log.read_text()


def test_proxy_credentials_redacted(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://testuser:abc123@127.0.0.1:8123")
    assert "abc123" not in module.redact("http://testuser:abc123@127.0.0.1:8123", module._redactions())
