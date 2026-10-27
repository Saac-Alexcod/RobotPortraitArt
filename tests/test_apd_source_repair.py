import sys
import hashlib
import json
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ensure_apdrawinggan_source as repair


@pytest.fixture(autouse=True)
def source_manifest(tmp_path, monkeypatch):
    data = b'bundled source'
    manifest = {'repository': 'example/project', 'revision': 'abc123',
                'files': {name: {'sha256': hashlib.sha256(data).hexdigest(), 'size': len(data)}
                          for name in repair.REQUIRED}}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest), encoding='utf-8')
    monkeypatch.setattr(repair, 'MANIFEST', path)
    return manifest


def test_restore_only_missing_source_preserves_weights_and_edits(tmp_path, monkeypatch):
    repo = tmp_path / repair.PREFIX
    repo.mkdir(parents=True)
    (repo / 'test.py').write_bytes(b'local edit')
    names = [repair.PREFIX + name for name in repair.REQUIRED]
    names.append(repair.PREFIX + 'checkpoints/model.pt')
    def git(args, **kwargs):
        if 'ls-tree' in args:
            return '\n'.join(names).encode()
        return b'bundled source'
    monkeypatch.setattr(repair.subprocess, 'check_output', git)
    repair.ensure_source(tmp_path)
    assert (repo / 'test.py').read_bytes() == b'local edit'
    assert (repo / 'models/test_model.py').read_bytes() == b'bundled source'
    assert not (repo / 'checkpoints/model.pt').exists()


@pytest.mark.parametrize('corrupt', [False, True])
def test_missing_source_without_git_downloads_verified_snapshot(tmp_path, monkeypatch, source_manifest, corrupt):
    def no_git(*a, **k):
        raise FileNotFoundError('git')
    monkeypatch.setattr(repair.subprocess, 'check_output', no_git)
    def download(url, target):
        assert url == 'https://codeload.github.com/example/project/zip/abc123'
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target, 'w') as bundle:
            for name in source_manifest['files']:
                bundle.writestr('project-abc123/' + repair.PREFIX + name,
                                b'corrupt source' if corrupt else b'bundled source')
            bundle.writestr('../../escape.py', b'ignored')
    monkeypatch.setattr(repair, 'download', download)
    if corrupt:
        with pytest.raises(ValueError, match='SHA256'):
            repair.ensure_source(tmp_path)
        assert not (tmp_path / repair.PREFIX / 'test.py').exists()
    else:
        repair.ensure_source(tmp_path)
        assert (tmp_path / repair.PREFIX / 'test.py').read_bytes() == b'bundled source'
        assert not (tmp_path / 'escape.py').exists()


def test_complete_archive_checkout_does_not_require_git(tmp_path, monkeypatch):
    for name in repair.REQUIRED:
        path = tmp_path / repair.PREFIX / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    def no_git(*a, **k):
        raise FileNotFoundError('git')
    monkeypatch.setattr(repair.subprocess, 'check_output', no_git)
    repair.ensure_source(tmp_path)
