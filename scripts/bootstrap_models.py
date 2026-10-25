"""Checked, opt-in downloads for the RobotPortraitArt optional model weights.

Uses Python standard library only. A network failure cannot overwrite existing files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'third_party' / 'models'
CHECKPOINT = ROOT / 'third_party' / 'APDrawingGAN' / 'checkpoints' / 'formal_author'
BASE = 'https://huggingface.co/opencv/opencv_zoo/resolve/main/models'
MODEL_INFO = {
    'yunet': {
        'name': 'face_detection_yunet_2023mar.onnx',
        'url': BASE + '/face_detection_yunet/face_detection_yunet_2023mar.onnx',
        'sha256': '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4',
    },
    'humanseg': {
        'name': 'human_segmentation_pphumanseg_2023mar.onnx',
        'url': BASE + '/human_segmentation_pphumanseg/human_segmentation_pphumanseg_2023mar.onnx',
        'sha256': '552d8a984054e59b5d773d24b9b12022b22046ceb2bbc4c9aaeaceb36a9ddf24',
    },
}
APD_URL = 'https://cg.cs.tsinghua.edu.cn/people/~Yongjin/APDrawingGAN-Models1.zip'
SPLIT_NAMES = ('G', 'GLEyel', 'GLEyer', 'GLNose', 'GLMouth', 'GLHair', 'GLBG', 'GCombine')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_present(directory: Path = CHECKPOINT) -> bool:
    combined = directory / '300_net_gen.pt'
    if combined.is_file() and combined.stat().st_size > 1024 * 1024:
        return True
    return all((directory / ('300_net_' + s + '.pth')).is_file() and
               (directory / ('300_net_' + s + '.pth')).stat().st_size > 65536
               for s in SPLIT_NAMES)


def check_models() -> dict[str, object]:
    result: dict[str, object] = {}
    for key, info in MODEL_INFO.items():
        file = MODELS / info['name']
        result[key] = {'path': str(file), 'ok': file.is_file() and sha256(file) == info['sha256']}
    result['apdrawinggan'] = {'path': str(CHECKPOINT), 'ok': checkpoint_present()}
    return result


def opener_for(proxy: str | None, no_proxy: bool) -> urllib.request.OpenerDirector:
    if proxy and no_proxy:
        raise ValueError('--proxy and --no-proxy are mutually exclusive')
    if proxy:
        parsed = urllib.parse.urlsplit(proxy)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname:
            raise ValueError('Proxy must be an http(s) URL (example: http://127.0.0.1:7890)')
        proxies = {'http': proxy, 'https': proxy}
    elif no_proxy:
        proxies = {}
    else:
        proxies = urllib.request.getproxies()
    return urllib.request.build_opener(urllib.request.ProxyHandler(proxies),
                                       urllib.request.HTTPSHandler(context=ssl.create_default_context()))


def download(url: str, destination: Path, opener: urllib.request.OpenerDirector,
             retries: int = 3, timeout: int = 35) -> Path:
    if urllib.parse.urlsplit(url).scheme != 'https':
        raise ValueError('Refusing non-HTTPS download URL')
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + '.part')
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {'User-Agent': 'RobotPortraitArt-Installer/1.0'}
            if offset:
                headers['Range'] = 'bytes=' + str(offset) + '-'
            request = urllib.request.Request(url, headers=headers)
            print(f'[MODEL] {destination.name}: attempt {attempt + 1}/{retries}; '
                  f'resumable bytes on disk={offset:,}', flush=True)
            with opener.open(request, timeout=timeout) as response:
                final_url = response.geturl()
                if urllib.parse.urlsplit(final_url).scheme != 'https':
                    raise ValueError('Redirected to insecure download URL')
                code = response.getcode()
                content_range = response.headers.get('Content-Range', '')
                resume = (offset > 0 and code == 206 and content_range.startswith('bytes ' + str(offset) + '-'))
                mode = 'ab' if resume else 'wb'
                if response.headers.get('Content-Type', '').lower().startswith('text/html'):
                    raise ValueError('Server returned HTML instead of model content')
                length_header = response.headers.get('Content-Length', '')
                length = int(length_header) if length_header.isdecimal() else 0
                total = length + (offset if resume else 0)
                start = time.monotonic()
                last_report = start
                bytes_now = offset if resume else 0
                with partial.open(mode) as output:
                    while True:
                        block = response.read(256 * 1024)
                        if not block:
                            break
                        output.write(block)
                        bytes_now += len(block)
                        now = time.monotonic()
                        if now - last_report >= 8 or (total and bytes_now >= total):
                            progress = (f'{bytes_now / (1024 ** 2):.1f} MiB / '
                                        f'{total / (1024 ** 2):.1f} MiB') if total else f'{bytes_now / (1024 ** 2):.1f} MiB'
                            print(f'[MODEL] {destination.name}: downloaded {progress}', flush=True)
                            last_report = now
                        if now - start > 6800:
                            raise TimeoutError('Model download exceeded 6800 seconds; partial file retained for retry')
                if length and bytes_now < total:
                    raise OSError(f'Truncated HTTP transfer: received {bytes_now} bytes, expected {total}')
                print(f'[MODEL] {destination.name}: network transfer finished '
                      f'({bytes_now / (1024 ** 2):.1f} MiB); validating...', flush=True)
            if partial.stat().st_size < 1024:
                raise ValueError('Downloaded response is unexpectedly small')
            return partial
        except (OSError, urllib.error.URLError, ValueError) as error:
            if isinstance(error, urllib.error.HTTPError) and error.code == 416:
                # The remote object changed (or partial is complete); restart safely.
                partial.unlink(missing_ok=True)
            last_error = error
            print(f'[MODEL WARNING] {destination.name}: {type(error).__name__}: {error}; '
                  f'attempt {attempt + 1}/{retries}. Partial download retained.',
                  file=sys.stderr, flush=True)
            if attempt + 1 < retries:
                time.sleep(min(2 ** attempt, 4))
    raise RuntimeError('Download failed after ' + str(retries) + ' attempts: ' + str(last_error))


def install_onnx(info: dict[str, str], opener: urllib.request.OpenerDirector,
                 *, url_override: str | None = None, retries: int = 3) -> str:
    target = MODELS / info['name']
    if target.is_file() and sha256(target) == info['sha256']:
        return 'already verified: ' + str(target)
    if target.exists():
        raise ValueError('Existing file failed SHA-256. Move it aside manually: ' + str(target))
    partial = download(url_override or info['url'], target, opener, retries=retries)
    actual = sha256(partial)
    if actual != info['sha256']:
        partial.unlink(missing_ok=True)
        raise ValueError('SHA-256 mismatch for ' + info['name'] + '; got ' + actual)
    os.replace(partial, target)
    return 'downloaded and verified: ' + str(target)


def extract_apd(zip_path: Path, dest: Path = CHECKPOINT) -> list[str]:
    """Extract ONLY whitelisted checkpoint names, never arbitrary archive paths."""
    if not zipfile.is_zipfile(zip_path):
        raise ValueError('APDrawingGAN download is not a ZIP archive')
    with zipfile.ZipFile(zip_path) as archive:
        entries = [entry for entry in archive.infolist() if not entry.is_dir()]
        combined = [x for x in entries if x.filename.replace('\\', '/').split('/')[-1] == '300_net_gen.pt']
        if len(combined) == 1:
            selected = [combined[0]]
        elif len(combined) > 1:
            raise ValueError('Ambiguous duplicated combined checkpoints in ZIP')
        else:
            selected = []
            for part in SPLIT_NAMES:
                name = '300_net_' + part + '.pth'
                matches = [x for x in entries if x.filename.replace('\\', '/').split('/')[-1] == name]
                if len(matches) != 1:
                    raise ValueError('Missing or ambiguous checkpoint in upstream ZIP: ' + name)
                selected.append(matches[0])
        if any(x.file_size < 1024 * 1024 for x in selected):
            raise ValueError('Upstream checkpoint appears too small')
        if sum(x.file_size for x in selected) > 3 * 1024**3:
            raise ValueError('Uncompressed model files exceed safety limit')
        dest.mkdir(parents=True, exist_ok=True)
        installed: list[str] = []
        if any((dest / x.filename.replace('\\', '/').split('/')[-1]).exists() for x in selected):
            raise ValueError('Refusing to overwrite existing checkpoint files')
        # Extract all entries to temp files first. Rename only once verification passes.
        try:
            for item in selected:
                name = item.filename.replace('\\', '/').split('/')[-1]
                target = dest / name
                if target.exists():
                    raise ValueError('Refusing to overwrite existing checkpoint: ' + str(target))
                part = dest / (name + '.part')
                installed.append(name)
                with archive.open(item) as source, part.open('wb') as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                if part.stat().st_size != item.file_size:
                    raise ValueError('Truncated extraction for ' + name)
            for name in installed:
                os.replace(dest / (name + '.part'), dest / name)
        except Exception:
            for name in installed:
                (dest / (name + '.part')).unlink(missing_ok=True)
            raise
    return installed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='Download both verified OpenCV ONNX models')
    parser.add_argument('--download-apd', action='store_true', help='Opt in to the official APDrawingGAN Model1 ZIP')
    parser.add_argument('--apd-sha256', help='Optional trusted SHA-256 for the upstream APDrawingGAN ZIP')
    parser.add_argument('--proxy', help='Optional HTTP(S) proxy for THIS process only')
    parser.add_argument('--no-proxy', action='store_true', help='Ignore proxy settings')
    parser.add_argument('--yunet-url', help='Alternative trusted HTTPS mirror (same checksum required)')
    parser.add_argument('--humanseg-url', help='Alternative trusted HTTPS mirror (same checksum required)')
    parser.add_argument('--retries', type=int, default=3)
    args = parser.parse_args(argv)
    if args.retries < 1 or args.retries > 10:
        parser.error('--retries must be between 1 and 10')
    if args.apd_sha256 and not re.fullmatch(r'[0-9a-fA-F]{64}', args.apd_sha256):
        parser.error('--apd-sha256 must be exactly 64 hex digits')
    print(json.dumps(check_models(), indent=2, ensure_ascii=False))
    if not args.download and not args.download_apd:
        return 0
    try:
        opener = opener_for(args.proxy, args.no_proxy)
        if args.download:
            for key in ('yunet', 'humanseg'):
                url = args.yunet_url if key == 'yunet' else args.humanseg_url
                print(install_onnx(MODEL_INFO[key], opener, url_override=url, retries=args.retries), flush=True)
        if args.download_apd:
            if checkpoint_present():
                print('APDrawingGAN checkpoint already present')
            else:
                # Never silently replace a previous download or an unknown checkpoint.
                target = CHECKPOINT / 'APDrawingGAN-Models1.zip'
                part = download(APD_URL, target, opener, retries=args.retries, timeout=90)
                if args.apd_sha256 and sha256(part).lower() != args.apd_sha256.lower():
                    part.unlink(missing_ok=True)
                    raise ValueError('APDrawingGAN Model1 ZIP SHA-256 mismatch')
                installed = extract_apd(part)
                part.unlink(missing_ok=True)
                print('APDrawingGAN checkpoints installed: ' + ', '.join(installed), flush=True)
        print('Final model status:', json.dumps(check_models(), ensure_ascii=False), flush=True)
        return 0
    except Exception as error:
        # Do not print proxy credentials and do not change global proxy settings.
        message = str(error)
        if args.proxy:
            message = message.replace(args.proxy, '<proxy-redacted>')
        print('MODEL SETUP FAILED: ' + message, file=sys.stderr)
        print('Check HTTPS proxy, DNS, certificate trust, firewall, and download source.', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
