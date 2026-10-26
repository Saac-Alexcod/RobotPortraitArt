"""Install an isolated Node 22 LTS into .rpa_tools without changing system Node.

Downloads only from nodejs.org over verified HTTPS. The official SHASUMS256
manifest must validate the exact ZIP before extraction or replacement.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VERSION = '22.16.0'
ARCHIVE = f'node-v{VERSION}-win-x64.zip'
PREFIX = f'node-v{VERSION}-win-x64/'
URL = f'https://nodejs.org/dist/v{VERSION}/'
TARGET = ROOT / '.rpa_tools' / f'node-v{VERSION}'


def manifest_sha256(manifest: str, filename: str) -> str:
    for line in manifest.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[1].lstrip('*') == filename:
            digest = fields[0].lower()
            if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('Invalid SHA256 format in manifest')
            return digest
    raise ValueError('Requested Node ZIP missing from official SHA256 manifest')


def safe_member(name: str) -> bool:
    if '\\' in name or ':' in name:
        return False
    path = PurePosixPath(name)
    return name.startswith(PREFIX) and all(p not in ('.', '..') for p in path.parts) \
        and not path.is_absolute()


def download_file(url: str, target: Path, *, proxy: str | None = None) -> None:
    if not url.startswith(URL) or not url.startswith('https://'):
        raise ValueError('Untrusted download URL')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler(
        {'http': proxy, 'https': proxy} if proxy else urllib.request.getproxies()))
    last_exc = None
    for attempt in range(3):
        try:
            with opener.open(url, timeout=45) as response, target.open('wb') as output:
                if not response.geturl().startswith('https://'):
                    raise ValueError('Insecure redirect')
                total = 0
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    total += len(block)
                    if total > 250 * 1024 * 1024:
                        raise ValueError('Node download exceeds expected size')
                    output.write(block)
                if total == 0:
                    raise ValueError('Empty download')
            return
        except (OSError, ValueError) as exc:
            last_exc = exc
            print(f'[RPA-E230] Node download attempt {attempt + 1} failed: {exc}', flush=True)
            if attempt != 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(f'Official Node download failed after retries: {last_exc}')


def install(proxy: str | None = None, target: Path = TARGET) -> Path:
    exe = target / 'node.exe'
    npm = target / 'npm.cmd'
    if exe.is_file() and npm.is_file():
        print(f'[RPA] Reusing portable Node at {target}', flush=True)
        return exe
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='rpa-node-', dir=target.parent) as tmp:
        tmpdir = Path(tmp)
        digest_file = tmpdir / 'manifest.txt'
        download_file(URL + 'SHASUMS256.txt', digest_file, proxy=proxy)
        expected = manifest_sha256(digest_file.read_text('utf-8'), ARCHIVE)
        archive = tmpdir / ARCHIVE
        download_file(URL + ARCHIVE, archive, proxy=proxy)
        actual = hashlib.sha256(archive.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Node ZIP SHA-256 mismatch; archive was not installed')
        staging = tmpdir / 'extracted'
        staging.mkdir()
        with zipfile.ZipFile(archive) as z:
            infos = z.infolist()
            if sum(x.file_size for x in infos) > 550 * 1024 * 1024:
                raise ValueError('Node archive uncompressed size exceeds safety limit')
            for info in infos:
                if not safe_member(info.filename):
                    raise ValueError('Unsafe Node ZIP entry: ' + info.filename)
                if info.is_dir():
                    continue
                # Reject symbolic links, not only path traversal.
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Unexpected ZIP symbolic link')
                relative = PurePosixPath(info.filename).relative_to(PREFIX.rstrip('/'))
                destination = staging.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as source, destination.open('wb') as dst:
                    shutil.copyfileobj(source, dst, length=1024 * 1024)
        if not (staging / 'node.exe').is_file() or not (staging / 'npm.cmd').is_file():
            raise ValueError('Official Node archive lacks node.exe or npm.cmd')
        if target.exists():
            raise RuntimeError('Existing incomplete portable Node directory: ' + str(target) +
                               '; move this directory to a backup before retrying')
        os.replace(staging, target)
    print(f'RPA_NODE={target / "node.exe"}', flush=True)
    return target / 'node.exe'


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--proxy', default=None)
    args = p.parse_args()
    try:
        install(args.proxy)
        return 0
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        print('[RPA-E230] ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
