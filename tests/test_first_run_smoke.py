"""Tests of the first-run inference gate; no CUDA/WSL or network required."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'verify_first_run.py'
SPEC = importlib.util.spec_from_file_location('verify_first_run', SCRIPT)
assert SPEC is not None and SPEC.loader is not None
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_missing_manifest_rejected(tmp_path: Path):
    ok, reason = mod.check_manifest(tmp_path)
    assert not ok and 'manifest' in reason


def test_baseline_manifest_cannot_pass_as_gan(tmp_path: Path):
    (tmp_path / 'manifest.json').write_text(json.dumps({'status': 'completed', 'method': 'baseline'}))
    ok, reason = mod.check_manifest(tmp_path)
    assert not ok and 'APDrawingGAN' in reason


def test_corrupt_image_rejected(tmp_path: Path):
    files = {'prepared': 'prepared.png', 'raster_portrait': 'portrait.png',
             'vector_svg': 'vector.svg', 'vector_render': 'render.png'}
    (tmp_path / 'manifest.json').write_text(json.dumps({'status': 'completed', 'method': 'apdrawinggan', 'files': files}))
    for filename in files.values():
        (tmp_path / filename).write_bytes(b'not real image' * 20)
    ok, reason = mod.check_manifest(tmp_path)
    assert not ok and 'invalid' in reason


def test_valid_artifacts_pass(tmp_path: Path):
    files = {'prepared': 'prepared.png', 'raster_portrait': 'portrait.png',
             'vector_svg': 'vector.svg', 'vector_render': 'render.png'}
    (tmp_path / 'manifest.json').write_text(json.dumps({'status': 'completed', 'method': 'apdrawinggan', 'files': files}))
    for key in ('prepared', 'raster_portrait', 'vector_render'):
        Image.new('RGB', (48, 48), (122, 125, 128)).save(tmp_path / files[key])
    (tmp_path / files['vector_svg']).write_text('<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0 L1 1"/></svg>')
    ok, reason = mod.check_manifest(tmp_path)
    assert ok, reason


def test_cartoon_requires_its_own_manifest_method(tmp_path: Path):
    files = {'prepared': 'prepared.png', 'raster_portrait': 'portrait.png',
             'vector_svg': 'vector.svg', 'vector_render': 'render.png'}
    (tmp_path / 'manifest.json').write_text(json.dumps({'status': 'completed', 'method': 'apdrawinggan', 'files': files}))
    ok, reason = mod.check_manifest(tmp_path, 'cartoon')
    assert not ok and 'cartoon' in reason


def test_traversal_name_rejected(tmp_path: Path):
    files = {'prepared': '../unsafe.png', 'raster_portrait': 'portrait.png',
             'vector_svg': 'vector.svg', 'vector_render': 'render.png'}
    (tmp_path / 'manifest.json').write_text(json.dumps({'status': 'completed', 'method': 'apdrawinggan', 'files': files}))
    ok, reason = mod.check_manifest(tmp_path)
    assert not ok and 'missing output' in reason


def test_invalid_onnx_is_backed_up_not_deleted(tmp_path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location('repair', SCRIPT.parent / 'repair_model_files.py')
    assert spec and spec.loader
    repair = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(repair)
    assets = tmp_path / 'models'
    assets.mkdir()
    one = next(iter(repair.MODEL_INFO.values()))['name']
    (assets / one).write_bytes(b'invalid model bytes')
    backup = tmp_path / 'backups'
    moved = repair.backup_invalid(models=assets, backup_dir=backup)
    assert len(moved) == 1
    assert moved[0].read_bytes() == b'invalid model bytes'
    assert not (assets / one).exists()
