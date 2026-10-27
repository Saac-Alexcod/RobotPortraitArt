"""Acceptance gate: generate an actual APDrawingGAN portrait and validate its artifacts.

This is intentionally NOT a model-preflight-only check.  On failure it exits nonzero.
The selected sample and report remain under ignored outputs/first_run_validation/.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import sys

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = (
    'portrait_sample.jpg',
    'portrait_glasses.jpg',
    'portrait_long_hair.jpg',
)


def check_manifest(run_dir: Path, method: str = 'apdrawinggan') -> tuple[bool, str]:
    path = run_dir / 'manifest.json'
    if not path.is_file():
        return False, 'missing manifest.json'
    try:
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if manifest.get('status') != 'completed' or manifest.get('method') != method:
            label = 'APDrawingGAN' if method == 'apdrawinggan' else method
            return False, f'manifest not completed using {label}'
        for key in ('prepared', 'raster_portrait', 'vector_svg', 'vector_render'):
            name = manifest['files'][key]
            if Path(name).name != name or not (run_dir / name).is_file():
                return False, 'missing output: ' + key
            minimum = 30 if key == 'vector_svg' else 100
            if (run_dir / name).stat().st_size < minimum:
                return False, 'empty/invalid output: ' + key
        from PIL import Image
        for image_key in ('raster_portrait', 'vector_render'):
            with Image.open(run_dir / manifest['files'][image_key]) as im:
                im.verify()
        ET.parse(run_dir / manifest['files']['vector_svg'])
        label = 'APDrawingGAN' if method == 'apdrawinggan' else method
        return True, f'completed {label} -> SVG -> render'
    except (ValueError, KeyError, OSError, TypeError, ET.ParseError) as exc:
        return False, f'invalid manifest: {exc}'


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--distro', default='Ubuntu-22.04')
    ap.add_argument('--python', required=True, dest='python_command')
    ap.add_argument('--method', choices=('apdrawinggan', 'cartoon'), default='apdrawinggan')
    ap.add_argument('--timeout', type=int, default=480)
    args = ap.parse_args(argv)
    out = ROOT / 'outputs' / 'first_run_validation' / args.method
    out.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for name in CANDIDATES:
        example = ROOT / 'data' / 'examples' / name
        result = out / example.stem
        cmd = [sys.executable, str(ROOT / 'scripts' / 'run_pipeline.py'),
               '--input', str(example), '--output-dir', str(result),
               '--method', args.method, '--config', str(ROOT / 'configs' / 'centerline.yaml'),
               '--size', '256' if args.method == 'cartoon' else '512',
               '--apd-distro', args.distro, '--apd-python', args.python_command]
        print('Attempting real GAN inference:', name, flush=True)
        try:
            process = subprocess.run(cmd, cwd=ROOT, text=True, encoding='utf-8', errors='replace',
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     timeout=args.timeout, check=False)
            (out / f'{example.stem}.log').write_text(process.stdout[-16000:], encoding='utf-8')
            ok, detail = check_manifest(result, args.method)
            if process.returncode == 0 and ok:
                print('[PASS]', detail, 'sample:', name, 'output:', result, flush=True)
                return 0
            failures.append(f'{name}: exit={process.returncode}, {detail}')
        except subprocess.TimeoutExpired:
            failures.append(f'{name}: timeout after {args.timeout} seconds')
    print(f'[FAILED] No example completed a real {args.method} pipeline.', file=sys.stderr)
    for error in failures:
        print(error, file=sys.stderr)
    print('See outputs/first_run_validation/*.log for the actual model errors.', file=sys.stderr)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
