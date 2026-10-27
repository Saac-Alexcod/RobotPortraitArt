"""Discover WSL CUDA and run APDrawingGAN and/or GANs N' Roses on a photo."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from start_web import ROOT, ensure_dependencies, inspect_apd, resolve_distro


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('all', 'apdrawinggan', 'cartoon'), default='all')
    parser.add_argument('--input', type=Path, default=ROOT / 'data/examples/portrait_sample.jpg')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/models')
    parser.add_argument('--distro', default=os.environ.get('ROBOT_PORTRAIT_APD_DISTRO', ''))
    parser.add_argument('--apd-python', default='')
    parser.add_argument('--setup', action='store_true', help='Install Windows/WSL dependencies and download selected models first')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.setup and args.apd_python:
        parser.error('--setup cannot be combined with --apd-python')
    if not args.input.is_file():
        parser.error(f'Input image not found: {args.input}')
    ensure_dependencies(check_only=not args.setup)
    distro = resolve_distro(args.distro)
    python = args.apd_python or ('' if args.setup else os.environ.get('ROBOT_PORTRAIT_APD_PYTHON', ''))
    report = inspect_apd(distro, python, args.setup)
    if not report['ready']:
        raise RuntimeError(report.get('error', 'WSL CUDA environment unavailable'))
    if args.setup:
        from download_models import setup_models
        setup_models(args.model)
    methods = ('apdrawinggan', 'cartoon') if args.model == 'all' else (args.model,)
    failed = []
    for method in methods:
        output = args.output_dir.resolve() / method
        command = [sys.executable, str(ROOT / 'scripts/run_pipeline.py'),
                   '--method', method, '--input', str(args.input.resolve()),
                   '--output-dir', str(output), '--size', '512' if method == 'apdrawinggan' else '256',
                   '--config', str(ROOT / 'configs/centerline.yaml'),
                   '--apd-distro', distro, '--apd-python', report['python'], '--seed', str(args.seed)]
        print(f'Running {method} -> {output}', flush=True)
        if subprocess.run(command, cwd=ROOT).returncode:
            failed.append(method)
    if failed:
        raise RuntimeError('Generation failed: ' + ', '.join(failed))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
