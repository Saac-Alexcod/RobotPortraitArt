"""Portable Node archive/manifest security checks (offline)."""
import importlib.util
from pathlib import Path
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'install_portable_node.py'
spec = importlib.util.spec_from_file_location('install_portable_node', SCRIPT)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_sha_manifest_selection():
    sha = 'a'*64
    assert mod.manifest_sha256(f'{sha}  other.zip\n{sha}  node.zip', 'node.zip') == sha
    with pytest.raises(ValueError):
        mod.manifest_sha256(f'{sha}  other.zip', 'node.zip')


def test_reject_unsafe_members():
    assert mod.safe_member(mod.PREFIX + 'node.exe')
    assert not mod.safe_member(mod.PREFIX + '../x.exe')
    assert not mod.safe_member('../node.exe')
    assert not mod.safe_member(mod.PREFIX + 'x\\evil.exe')
