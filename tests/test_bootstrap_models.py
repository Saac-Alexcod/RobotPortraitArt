"""Offline tests: model bootstrap never touches the network in this suite."""
import hashlib
import io
from pathlib import Path
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import bootstrap_models as bm


class Response(io.BytesIO):
    headers = {'Content-Type': 'application/octet-stream'}
    def __init__(self, contents: bytes, url: str, code: int = 200):
        super().__init__(contents)
        self.url, self.code = url, code
    def geturl(self):
        return self.url
    def getcode(self):
        return self.code
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.close()


class Opener:
    def __init__(self, contents: bytes):
        self.contents, self.calls = contents, []
    def open(self, req, timeout=35):
        self.calls.append(req)
        return Response(self.contents, req.full_url)


def test_models_installed_atomically_and_checked(tmp_path, monkeypatch):
    monkeypatch.setattr(bm, 'MODELS', tmp_path)
    model = b'FAKE-ONNX-FOR-TEST' * 100
    info = {'name': 'yunet.onnx', 'url': 'https://example.org/model',
            'sha256': hashlib.sha256(model).hexdigest()}
    opener = Opener(model)
    assert 'verified' in bm.install_onnx(info, opener)
    assert (tmp_path / 'yunet.onnx').read_bytes() == model
    assert not (tmp_path / 'yunet.onnx.part').exists()
    assert 'already verified' in bm.install_onnx(info, opener)
    assert len(opener.calls) == 1


def test_bad_download_never_installs_or_clobbers(tmp_path, monkeypatch):
    monkeypatch.setattr(bm, 'MODELS', tmp_path)
    info = {'name': 'yunet.onnx', 'url': 'https://example.org/model', 'sha256': '0' * 64}
    with pytest.raises(ValueError, match='mismatch'):
        bm.install_onnx(info, Opener(b'A' * 2000))
    assert not (tmp_path / 'yunet.onnx').exists()
    (tmp_path / 'yunet.onnx').write_bytes(b'keep-me')
    with pytest.raises(ValueError, match='Existing file'):
        bm.install_onnx(info, Opener(b'A' * 2000))
    assert (tmp_path / 'yunet.onnx').read_bytes() == b'keep-me'


def test_resume_refused_when_server_does_not_honor_range(tmp_path):
    dest = tmp_path / 'model.bin'
    (tmp_path / 'model.bin.part').write_bytes(b'oldbytes')
    bm.download('https://example.org/model.bin', dest, Opener(b'new' * 500))
    assert (tmp_path / 'model.bin.part').read_bytes() == b'new' * 500


def test_proxy_validation_and_insecure_url(tmp_path):
    with pytest.raises(ValueError, match='Proxy'):
        bm.opener_for('socks5://localhost:9000', False)
    with pytest.raises(ValueError, match='mutually'):
        bm.opener_for('http://127.0.0.1:7890', True)
    with pytest.raises(ValueError, match='non-HTTPS'):
        bm.download('http://example.org/a', tmp_path / 'a', Opener(b'A' * 2000))


def test_apd_whitelisted_extract_and_refuses_overwrite(tmp_path):
    source = tmp_path / 'model.zip'
    with zipfile.ZipFile(source, 'w') as z:
        z.writestr('dir/300_net_gen.pt', b'P' * (1024 * 1024 + 100))
        z.writestr('../../malicious.txt', b'bad')
    dest = tmp_path / 'out'
    assert bm.extract_apd(source, dest) == ['300_net_gen.pt']
    assert (dest / '300_net_gen.pt').is_file()
    assert not (tmp_path / 'malicious.txt').exists()
    with pytest.raises(ValueError, match='overwrite'):
        bm.extract_apd(source, dest)


def test_apd_invalid_zip_and_incomplete_split(tmp_path):
    source = tmp_path / 'model.zip'
    source.write_bytes(b'<html>error</html>')
    with pytest.raises(ValueError, match='not a ZIP'):
        bm.extract_apd(source, tmp_path / 'out')
    with zipfile.ZipFile(source, 'w') as z:
        z.writestr('300_net_G.pth', b'P' * (1024 * 1024 + 10))
    with pytest.raises(ValueError, match='Missing'):
        bm.extract_apd(source, tmp_path / 'out')
