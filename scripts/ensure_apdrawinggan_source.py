"""Restore or download the pinned, project-compatible APDrawingGAN source."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from model_download import download

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name('apdrawinggan_source_manifest.json')
PREFIX = 'third_party/APDrawingGAN/'
REQUIRED = ('test.py', 'options/base_options.py', 'options/test_options.py',
            'data/single_dataset.py', 'data/base_dataset.py', 'data/image_folder.py',
            'models/test_model.py', 'models/base_model.py', 'models/networks.py',
            'util/util.py', 'util/html.py', 'util/visualizer.py')


def download_source(root, manifest, missing):
    revision = manifest['revision']
    url = f"https://codeload.github.com/{manifest['repository']}/zip/{revision}"
    archive = root / '.cache/models' / f'apdrawinggan-source-{revision}.zip'
    download(url, archive)
    try:
        with zipfile.ZipFile(archive) as bundle:
            archive_root = manifest['repository'].split('/')[-1] + '-' + revision
            payloads = {}
            # Read only exact manifest entries, verify all before installing any.
            for name in missing:
                info = manifest['files'][name]
                member = f"{archive_root}/{PREFIX}{name}"
                if bundle.getinfo(member).file_size != info['size']:
                    raise ValueError(f'APDrawingGAN source size mismatch: {name}')
                data = bundle.read(member)
                if hashlib.sha256(data).hexdigest() != info['sha256']:
                    raise ValueError(f'APDrawingGAN source SHA256 mismatch: {name}')
                payloads[name] = data
    except (zipfile.BadZipFile, KeyError, ValueError):
        archive.unlink(missing_ok=True)
        raise
    for name, data in payloads.items():
        target = root / PREFIX / name
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as output:
            output.write(data)
        print(f'Downloaded source: {name}', flush=True)


def ensure_source(root=ROOT):
    root = Path(root)
    repo = root / PREFIX
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    names = list(manifest['files'])
    for name in names:
        if Path(name).is_absolute() or '..' in Path(name).parts or '\\' in name or ':' in name:
            raise ValueError(f'Invalid source manifest path: {name}')
    missing = [name for name in names if not (repo / name).is_file()]
    if not missing:
        print('APDrawingGAN bundled source ready.', flush=True)
        return
    try:
        tracked = subprocess.check_output(
            ['git', '-C', str(root), 'ls-tree', '-r', '--name-only', 'HEAD', '--', PREFIX],
            stderr=subprocess.PIPE, timeout=30).decode('utf-8').splitlines()
    except (OSError, subprocess.SubprocessError):
        tracked = []
    for name in tracked:
        # Only restore bundled source; never restore generated model artifacts.
        relative = Path(name)
        if not name.startswith(PREFIX) or '..' in relative.parts:
            continue
        if any(part in ('checkpoints', 'results', 'dataset') for part in relative.parts):
            continue
        target = root / relative
        if target.exists():
            continue
        content = subprocess.check_output(['git', '-C', str(root), 'show', f'HEAD:{name}'], timeout=30)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(content)
        print(f'Restored bundled source: {name}', flush=True)
    missing = [name for name in names if not (repo / name).is_file()]
    if missing:
        print('Downloading missing APDrawingGAN source from pinned project snapshot...', flush=True)
        download_source(root, manifest, missing)
    print('APDrawingGAN bundled source ready.', flush=True)


if __name__ == '__main__':
    ensure_source()
