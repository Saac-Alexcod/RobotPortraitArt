"""Safely back up a known invalid ONNX file so bootstrap_models can re-download it.

Never overwrites or deletes a user-provided file. Only handles two pinned
third-party ONNX assets. Unknown checkpoint files are not touched.
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap_models import MODEL_INFO, MODELS, sha256


def backup_invalid(*, models: Path = MODELS, backup_dir: Path | None = None) -> list[Path]:
    if backup_dir is None:
        backup_dir = Path(__file__).resolve().parents[1] / 'outputs' / 'invalid_model_backups'
    results: list[Path] = []
    for info in MODEL_INFO.values():
        path = models / info['name']
        if not path.exists():
            continue
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'Unexpected existing model path: {path}')
        if sha256(path) == info['sha256']:
            continue
        backup_dir.mkdir(parents=True, exist_ok=True)
        suffix = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        backup = backup_dir / f'{path.name}.{suffix}.invalid'
        n = 0
        while backup.exists():
            n += 1
            backup = backup_dir / f'{path.name}.{suffix}.{n}.invalid'
        path.replace(backup)
        print(f'[BACKUP] Invalid ONNX moved to {backup}', flush=True)
        results.append(backup)
    return results


if __name__ == '__main__':
    backup_invalid()
