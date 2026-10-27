"""Read-only Windows/WSL/PyTorch/CUDA and model diagnostics for RobotPortraitArt."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from bootstrap_models import check_models  # noqa: E402


def run(args: list[str], *, source: str | None = None, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, input=source, capture_output=True, text=True,
                          encoding='utf-8', errors='replace', check=False, timeout=timeout)


def inspect(*, distro: str = 'Ubuntu-22.04', python_cmd: str | None = None) -> dict[str, object]:
    win_python = ROOT / '.venv' / 'Scripts' / 'python.exe'
    report: dict[str, object] = {
        'windows_venv_exists': win_python.is_file(),
        'windows_python_running': sys.executable,
        'windows_backend_modules': {x: importlib.util.find_spec(x) is not None
                                    for x in ('fastapi', 'uvicorn', 'cv2')},
        'models': check_models(),
        'wsl': {'available': False, 'distro': distro},
        'recommendations': [],
    }
    advice: list[str] = report['recommendations']  # type: ignore[assignment]
    if not win_python.is_file():
        advice.append('Create .venv with a compatible Windows Python and install requirements-web.txt.')
    models = report['models']
    if any(not x['ok'] for x in models.values()):  # type: ignore[union-attr]
        advice.append('Run scripts/setup_environment.ps1 -DownloadModels -DownloadAPD (opt-in downloads).')
    wsl = shutil.which('wsl.exe')
    if not wsl:
        advice.append('WSL is unavailable. Use Baseline or install WSL2 following Microsoft documentation.')
        return report
    try:
        # Verify the selected distro, rather than just the existence of wsl.exe.
        probe = run([wsl, '-d', distro, '--', 'sh', '-lc', 'printf WSL_OK'], timeout=25)
        if probe.returncode or 'WSL_OK' not in probe.stdout:
            report['wsl'] = {'available': False, 'distro': distro,
                             'error': (probe.stderr or probe.stdout).strip()[:500]}
            advice.append('Check wsl --list --verbose and choose an installed WSL2 distro.')
            return report
        report['wsl'] = {'available': True, 'distro': distro}
        if not python_cmd:
            discover = ('p="$HOME/.venvs/robotportrait-apd/bin/python"; '
                        'if [ -x "$p" ]; then printf "%s" "$p"; '
                        'else command -v python3 || true; fi')
            found = run([wsl, '-d', distro, '--', 'sh', '-s'], source=discover)
            python_cmd = found.stdout.strip()
        if not python_cmd:
            report['wsl']['python'] = {'available': False}  # type: ignore[index]
            advice.append('Install Python3/venv inside WSL2 and create an APDrawingGAN environment.')
            return report
        code = '''import json, sys
result = {'executable':sys.executable, 'version':sys.version.split()[0], 'torch':False, 'torchvision':False, 'cuda':False}
for name in ('torch','torchvision'):
 try:
  x = __import__(name)
  result[name] = True
  result[name+'_version'] = str(getattr(x, '__version__', 'unknown'))
 except Exception as error:
  result[name+'_error'] = str(error)[:250]
if result['torch']:
 try:
  import torch
  result['cuda'] = bool(torch.cuda.is_available())
  if result['cuda']: result['gpu'] = torch.cuda.get_device_name(0)
 except Exception as error:
  result['cuda_error'] = str(error)[:250]
print('RPA_DOCTOR_JSON=' + json.dumps(result))
'''
        # Use stdin: it avoids PowerShell/wsl argument quote stripping.
        checked = run([wsl, '-d', distro, '--', python_cmd, '-'], source=code, timeout=45)
        lines = [x.split('=', 1)[1] for x in checked.stdout.splitlines()
                 if x.startswith('RPA_DOCTOR_JSON=')]
        if checked.returncode or not lines:
            report['wsl']['python'] = {'available': False, 'command': python_cmd,
                                       'error': (checked.stderr or checked.stdout).strip()[:500]}  # type: ignore[index]
            advice.append('Selected WSL Python failed. Pass --python with the actual virtualenv interpreter.')
        else:
            runtime = json.loads(lines[-1])
            runtime['available'] = True
            report['wsl']['python'] = runtime  # type: ignore[index]
            if not runtime['torch'] or not runtime['torchvision']:
                advice.append('Install compatible torch/torchvision in the WSL virtualenv; use the official PyTorch selector.')
            elif not runtime['cuda']:
                advice.append('CUDA unavailable to WSL PyTorch. Check Windows NVIDIA driver, WSL2, and CUDA-enabled torch build.')
    except (OSError, subprocess.TimeoutExpired) as exc:
        report['wsl']['error'] = str(exc)[:300]  # type: ignore[index]
        advice.append('WSL check failed/timed out. Inspect distro and interpreter manually.')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--distro', default='Ubuntu-22.04')
    parser.add_argument('--python', default=None)
    parser.add_argument('--json', default=None, help='Optional report output path')
    args = parser.parse_args()
    report = inspect(distro=args.distro, python_cmd=args.python)
    result = json.dumps(report, ensure_ascii=False, indent=2)
    print(result)
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result + '\n', encoding='utf-8')
    print('\nStatus: ' + ('READY CANDIDATE' if (all(x['ok'] for x in report['models'].values())
          and report['wsl'].get('python', {}).get('cuda')) else 'SETUP REQUIRED'))
    print('Note: READY CANDIDATE is not a substitute for a real APDrawingGAN inference test.')


if __name__ == '__main__':
    main()
