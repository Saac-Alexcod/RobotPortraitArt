"""Streaming, validated, atomic model downloads (standard library only)."""
from pathlib import Path
import hashlib
import time
import urllib.request


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, destination, *, expected_size=None, expected_sha256=None):
    destination = Path(destination)
    def validate(path):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'Empty or missing file: {path}')
        if expected_size is not None and path.stat().st_size != expected_size:
            raise ValueError(f'Unexpected size: {path}')
        digest = sha256(path)
        if expected_sha256 and digest != expected_sha256:
            raise ValueError(f'SHA256 mismatch: {path}')
        return digest
    if destination.is_file():
        try:
            return validate(destination)
        except ValueError:
            print(f'Repairing invalid file: {destination}', flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.download')
    for attempt in range(3):
        try:
            print(f'Downloading {destination.name} (attempt {attempt + 1}/3)...', flush=True)
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as output:
                if 'text/html' in response.headers.get('Content-Type', ''):
                    raise ValueError(f'Server returned HTML instead of model: {url}')
                for chunk in iter(lambda: response.read(1024 * 1024), b''):
                    output.write(chunk)
            digest = validate(temporary)
            temporary.replace(destination)
            return digest
        except (OSError, ValueError):
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
