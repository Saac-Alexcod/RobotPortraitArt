"""Offline compatibility tests; no system mutations or WSL required."""
import importlib.util
from pathlib import Path
import subprocess

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'installer_compatibility.py'
spec = importlib.util.spec_from_file_location('installer_compatibility', SCRIPT)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_wsl_utf16_list_and_leading_star():
    raw = '\ufeffUbuntu-24.04\r\nUbuntu-22.04\r\n'.encode('utf-16-le')
    assert mod.distro_names(raw) == ['Ubuntu-24.04', 'Ubuntu-22.04']


def test_versions_are_guarded():
    assert mod.compatible_python('3.11.9')
    assert mod.compatible_python('3.12')
    assert not mod.compatible_python('3.13')
    assert not mod.compatible_python('3.8')


def test_auto_selects_working_other_ubuntu(monkeypatch):
    monkeypatch.setattr(mod, '_run', lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, 'Ubuntu-24.04\n'.encode(), b''))
    monkeypatch.setattr(mod, 'probe_distro', lambda name: {'name': name, 'usable': True,
        'python_version': '3.12', 'runtime_ready': False})
    state = mod.inspect()
    assert state['status'] == 'compatible'
    assert state['selected']['name'] == 'Ubuntu-24.04'


def test_wsl_registered_incompatible_not_marked_missing(monkeypatch):
    monkeypatch.setattr(mod, '_run', lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, b'Ubuntu-26.04\n', b''))
    monkeypatch.setattr(mod, 'probe_distro', lambda name: {'name': name, 'usable': False,
        'python_version': '3.13', 'reason': 'Python 3.13 not supported'})
    result = mod.inspect()
    assert result['status'] == 'needs_fallback'
    assert result['registered'] == ['Ubuntu-26.04']


def test_error_listing_is_not_no_install(monkeypatch):
    monkeypatch.setattr(mod, '_run', lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1, b'', b'WSL broken'))
    assert mod.inspect()['status'] == 'wsl_error'


def test_auto_prefers_working_runtime(monkeypatch):
    monkeypatch.setattr(mod, '_run', lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, b'Ubuntu-22.04\nUbuntu-24.04\n', b''))
    def fake(name):
        return {'name': name, 'usable': True, 'runtime_ready': name == 'Ubuntu-24.04'}
    monkeypatch.setattr(mod, 'probe_distro', fake)
    assert mod.inspect()['selected']['name'] == 'Ubuntu-24.04'


def test_explicit_distro_not_silently_ignored(monkeypatch):
    monkeypatch.setattr(mod, '_run', lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, b'Ubuntu-22.04\nUbuntu-24.04\n', b''))
    monkeypatch.setattr(mod, 'probe_distro', lambda name: {'name': name, 'usable': True, 'runtime_ready': True})
    assert mod.inspect('Ubuntu-22.04')['selected']['name'] == 'Ubuntu-22.04'


def test_ubuntu_2404_without_python_can_be_bootstrapped(monkeypatch):
    original = mod.LINUX_PROBE
    assert 'RPA_DISCOVER|' in original
    assert "os_release in ('22.04', '24.04')" in __import__('inspect').getsource(mod.probe_distro)
    # Simulated successful WSL2 Ubuntu-24.04 probe with apt but no Python installed.
    fake = subprocess.CompletedProcess(['wsl.exe'], 0,
                                       b'RPA_DISCOVER|yes|ubuntu|24.04|||no\n', b'')
    monkeypatch.setattr(mod, '_run', lambda *args, **kwargs: fake)
    assert mod.probe_distro('Ubuntu-24.04')['usable']


def test_wsl1_not_accepted_as_cuda_runtime(monkeypatch):
    fake = subprocess.CompletedProcess(['wsl.exe'], 0,
                                       b'RPA_DISCOVER|no|ubuntu|22.04|3.10|python3|yes\n', b'')
    monkeypatch.setattr(mod, '_run', lambda *args, **kwargs: fake)
    assert not mod.probe_distro('Ubuntu-22.04')['usable']


def test_no_registered_distributions_reports_install_candidate(monkeypatch):
    monkeypatch.setattr(mod, '_run', lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1, b'', b'No installed distributions'))
    assert mod.inspect()['status'] == 'not_installed'
