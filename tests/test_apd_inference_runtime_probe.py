"""Regressions for incorrect Visdom requirements in first-run installer."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import types

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "apd_runtime_probe.py"


def load_probe():
    spec = importlib.util.spec_from_file_location("rpa_apd_inference_probe", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_visdom_is_not_in_required_inference_modules():
    probe = load_probe()
    assert "visdom" not in probe.REQUIRED_INFERENCE_MODULES
    assert "cv2" not in probe.REQUIRED_INFERENCE_MODULES


def test_probe_succeeds_without_visdom(monkeypatch, capsys):
    probe = load_probe()
    fake_cuda = types.SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: "Mock GPU")
    class FakeTensor:
        def __add__(self, other):
            return self
        def item(self):
            return 2.0
    fake_torch = types.SimpleNamespace(__version__="2.5.1", cuda=fake_cuda,
                                       ones=lambda *args, **kwargs: FakeTensor())
    def importing(name):
        assert name != "visdom", "training-only module must never be imported"
        return fake_torch if name == "torch" else types.SimpleNamespace()
    monkeypatch.setattr(probe.importlib, "import_module", importing)
    assert probe.check_runtime() == 0
    assert "RPA_RUNTIME_OK" in capsys.readouterr().out


def test_probe_reports_missing_required_dependency(monkeypatch, capsys):
    probe = load_probe()
    def importing(name):
        if name == "dominate":
            raise ModuleNotFoundError("No module named 'dominate'")
        return types.SimpleNamespace()
    monkeypatch.setattr(probe.importlib, "import_module", importing)
    assert probe.check_runtime() == 21
    assert "dominate" in capsys.readouterr().err


def test_probe_reports_cuda_unavailable(monkeypatch, capsys):
    probe = load_probe()
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr(probe.importlib, "import_module", lambda name: fake_torch if name == "torch" else object())
    assert probe.check_runtime() == 22
    assert "CUDA" in capsys.readouterr().err


def test_all_installer_probes_match_inference_not_training():
    ps = (ROOT / "install.ps1").read_text(encoding="utf-8")
    compat = (ROOT / "scripts/installer_compatibility.py").read_text(encoding="utf-8")
    sh = (ROOT / "scripts/install_wsl_runtime.sh").read_text(encoding="utf-8")
    assert "InputText $runtimeProbe" in ps
    assert "visdom; assert" not in ps
    assert "cv2,visdom" not in compat
    assert "cv2,visdom" not in sh
    assert "RPA_RUNTIME_OK" in (ROOT / "scripts/apd_runtime_probe.py").read_text(encoding="utf-8")
