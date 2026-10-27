"""Offline regression tests for installation integrity and model dispatch."""
import hashlib
import io
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import model_download
import run_models
import setup_apdrawinggan as apd


def test_download_repairs_corruption_and_reuses_valid_file(tmp_path, monkeypatch):
    target = tmp_path / 'model.pt'
    target.write_bytes(b'bad')
    content = b'valid weights'
    digest = hashlib.sha256(content).hexdigest()
    class Response(io.BytesIO):
        headers = {'Content-Type': 'application/octet-stream'}
    monkeypatch.setattr(model_download.urllib.request, 'urlopen', lambda *a, **k: Response(content))
    assert model_download.download('https://example.test', target, expected_sha256=digest) == digest
    monkeypatch.setattr(model_download.urllib.request, 'urlopen', lambda *a, **k: pytest.fail('cache should be reused'))
    assert model_download.download('https://example.test', target, expected_sha256=digest) == digest


def test_failed_download_preserves_existing_file(tmp_path, monkeypatch):
    target = tmp_path / 'model.pt'
    target.write_bytes(b'original')
    class Response(io.BytesIO):
        headers = {'Content-Type': 'text/html'}
    monkeypatch.setattr(model_download.urllib.request, 'urlopen', lambda *a, **k: Response(b'login page'))
    monkeypatch.setattr(model_download.time, 'sleep', lambda *_: None)
    with pytest.raises(ValueError, match='HTML'):
        model_download.download('https://example.test', target, expected_sha256='invalid')
    assert target.read_bytes() == b'original'
    assert not target.with_name('model.pt.download').exists()


def test_apd_extracts_only_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(apd, 'ROOT', tmp_path)
    monkeypatch.setattr(apd, 'ONNX', {})
    content = b'checkpoint'
    monkeypatch.setattr(apd, 'MODEL_SHA256', hashlib.sha256(content).hexdigest())
    repo = tmp_path / 'third_party/APDrawingGAN'
    repo.mkdir(parents=True)
    (repo / 'test.py').touch()
    archive = tmp_path / 'APDrawingGAN-Models1.zip'
    with zipfile.ZipFile(archive, 'w') as bundle:
        bundle.writestr('formal_author/300_net_gen.pt', content)
        bundle.writestr('../../outside.txt', b'unsafe')
    monkeypatch.setattr(apd, 'ARCHIVE_SHA256', model_download.sha256(archive))
    apd.setup()
    assert (repo / 'checkpoints/formal_author/300_net_gen.pt').read_bytes() == content
    assert not (tmp_path / 'outside.txt').exists()


def test_run_both_uses_correct_sizes_and_reports_failure(tmp_path, monkeypatch):
    photo = tmp_path / 'portrait with spaces.jpg'
    photo.touch()
    monkeypatch.setattr(sys, 'argv', ['run_models.py', '--input', str(photo)])
    monkeypatch.setattr(run_models, 'ensure_dependencies', lambda **k: None)
    monkeypatch.setattr(run_models, 'resolve_distro', lambda *_: 'Ubuntu')
    monkeypatch.setattr(run_models, 'inspect_apd', lambda *a: {'ready': True, 'python': '/env/python'})
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 1 if len(commands) == 1 else 0)
    monkeypatch.setattr(run_models.subprocess, 'run', run)
    with pytest.raises(RuntimeError, match='apdrawinggan'):
        run_models.main()
    assert [c[c.index('--size') + 1] for c in commands] == ['512', '256']
    assert all(c[c.index('--input') + 1] == str(photo) for c in commands)
