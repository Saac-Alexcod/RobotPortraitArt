"""Windows first-run WSL discovery and compatibility policy (read-only).

Prefer *working and compatible* existing WSL2 installations. Never interpret a
failed probe as proof that a registered distribution is missing. Writes JSON so
PowerShell never needs to parse WSL's UTF-16 console output.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# Verified installation candidate: PyTorch 2.5.1 / torchvision 0.20.1.
# A Python version within this range is necessary, NOT sufficient, for working GAN.
PY_MIN = (3, 9)
PY_MAX = (3, 12)
DISTRO_PREFERENCE = ('Ubuntu-22.04', 'Ubuntu-24.04', 'Ubuntu')


def decode_wsl(data: bytes) -> str:
    if data.startswith((b'\xff\xfe', b'\xfe\xff')):
        return data.decode('utf-16', errors='replace')
    if len(data) >= 4 and data[1::2].count(0) > len(data) // 8:
        return data.decode('utf-16-le', errors='replace')
    return data.decode('utf-8', errors='replace')


def distro_names(raw: bytes) -> list[str]:
    value = decode_wsl(raw).replace('\ufeff', '').replace('\x00', '')
    return list(dict.fromkeys(line.strip().lstrip('*').strip() for line in value.splitlines()
                              if line.strip().strip('*').strip()))


def compatible_python(value: str) -> bool:
    match = re.match(r'^\s*(\d+)\.(\d+)(?:\.\d+)?\s*$', value)
    return bool(match and PY_MIN <= (int(match[1]), int(match[2])) <= PY_MAX)


def distro_rank(name: str) -> int:
    if name in DISTRO_PREFERENCE:
        return DISTRO_PREFERENCE.index(name)
    if name.startswith('Ubuntu-'):
        return 10
    return 20


LINUX_PROBE = r'''#!/bin/sh
set +e
kernel=$(uname -r 2>/dev/null)
case "$kernel" in *[Ww][Ss][Ll]2*|*[Mm]icrosoft-standard*) wsl2=yes ;; *) wsl2=no ;; esac
id=unknown
release=unknown
if [ -r /etc/os-release ]; then
  id=$(sed -n 's/^ID=//p' /etc/os-release | head -1 | tr -d '"')
  release=$(sed -n 's/^VERSION_ID=//p' /etc/os-release | head -1 | tr -d '"')
fi
selected=''
version=''
ready='no'
# Do not replace an already working venv; accept its exact Python version.
for py in "$HOME/.venvs/robotportrait-apd/bin/python" python3.11 python3.10 python3.12 python3.9 python3; do
  if ! command -v "$py" >/dev/null 2>&1 && [ ! -x "$py" ]; then continue; fi
  v=$($py -c 'import sys; print(str(sys.version_info.major)+"."+str(sys.version_info.minor))' 2>/dev/null)
  case "$v" in 3.9|3.10|3.11|3.12) ;; *) continue ;; esac
  if [ -z "$selected" ]; then selected="$py"; version="$v"; fi
  if $py -c 'import torch,torchvision,dominate,numpy,PIL; assert torch.cuda.is_available()' >/dev/null 2>&1; then
    selected="$py"; version="$v"; ready=yes; break
  fi
done
printf 'RPA_DISCOVER|%s|%s|%s|%s|%s|%s\n' "$wsl2" "$id" "$release" "$version" "$selected" "$ready"
'''


def _run(command: list[str], timeout: int, source: str | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(command, input=source.encode() if source is not None else None,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
                          check=False)


def probe_distro(name: str) -> dict[str, object]:
    try:
        p = _run(['wsl.exe', '-d', name, '--', 'sh', '-s'], 60, LINUX_PROBE)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'name': name, 'usable': False, 'reason': str(exc)[:180]}
    lines = [x for x in decode_wsl(p.stdout).splitlines() if x.startswith('RPA_DISCOVER|')]
    if p.returncode or not lines:
        return {'name': name, 'usable': False,
                'reason': (decode_wsl(p.stderr) or decode_wsl(p.stdout)).strip()[:180]}
    fields = lines[-1].split('|')
    if len(fields) != 7:
        return {'name': name, 'usable': False, 'reason': 'Malformed probe result'}
    _, wsl2, os_id, os_release, version, py, ready = fields
    supported = compatible_python(version)
    # Ubuntu 22.04/24.04 can install the compatible default Python with apt.
    installable = os_id == 'ubuntu' and os_release in ('22.04', '24.04')
    return {'name': name, 'wsl2': wsl2 == 'yes', 'os_id': os_id, 'os_release': os_release,
            'python': py, 'python_version': version, 'runtime_ready': ready == 'yes',
            'usable': wsl2 == 'yes' and (supported or installable) and
                      (ready == 'yes' or os_id in ('ubuntu', 'debian')),
            'reason': '' if wsl2 == 'yes' and (supported or installable) else
                      ('WSL1 is not suitable for CUDA' if wsl2 != 'yes' else
                       'No Python 3.9-3.12 interpreter in this distribution')}


def inspect(preferred: str = '') -> dict[str, object]:
    try:
        p = _run(['wsl.exe', '--list', '--quiet'], 25)
    except FileNotFoundError:
        return {'status': 'wsl_missing', 'registered': [], 'selected': None,
                'help': 'wsl.exe is unavailable; check Windows optional features and virtualization'}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'status': 'wsl_error', 'registered': [], 'selected': None, 'help': str(exc)}
    if p.returncode:
        explanation = decode_wsl(p.stderr) + decode_wsl(p.stdout)
        if any(word in explanation.lower() for word in
               ('no installed distributions', 'there are no installed distributions',
                '没有已安装', '没有安装任何', '未安装任何发行版')):
            return {'status': 'not_installed', 'registered': [], 'selected': None,
                    'probes': [], 'help': 'No registered WSL distributions; install Ubuntu-22.04.'}
        return {'status': 'wsl_error', 'registered': [], 'selected': None,
                'help': (decode_wsl(p.stderr) or decode_wsl(p.stdout)).strip()[:300]}
    installed = distro_names(p.stdout)
    candidates = sorted(installed, key=lambda name: (name != preferred if preferred else False,
                                                    distro_rank(name)))
    observations = [probe_distro(name) for name in candidates]
    matches = [x for x in observations if x.get('usable') and (not preferred or x['name'] == preferred)]
    # A ready environment is preferable to a Python that needs package installation.
    matches.sort(key=lambda x: (not x['runtime_ready'], distro_rank(str(x['name']))))
    picked = matches[0] if matches else None
    status = ('ready' if picked and picked['runtime_ready'] else 'compatible') if picked else (
        'not_installed' if not installed or (preferred and preferred not in installed)
        else 'needs_fallback')
    return {'status': status, 'registered': installed, 'selected': picked,
            'probes': observations, 'requested': preferred,
            'help': 'Existing distro cannot be reused; preserve it and install compatible Ubuntu-22.04 alongside.'
                    if status == 'needs_fallback' else ''}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefer', default='', help='Explicit WSL distribution (auto-detect if empty)')
    parser.add_argument('--json', type=Path, help='Write machine-readable report')
    args = parser.parse_args(argv)
    report = inspect(args.prefer)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print('RPA_WSL_SELECTION=' + str(report['status']), flush=True)
    if report['selected']:
        print('RPA_SELECTED_DISTRO=' + str(report['selected']['name']), flush=True)
    for row in report.get('probes', []):
        print('RPA_DISTRO=' + str(row['name']) + ' ready=' + str(row.get('runtime_ready', False)) +
              ' python=' + str(row.get('python_version', 'missing')) +
              ' usable=' + str(row.get('usable', False)), flush=True)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + '\n', encoding='utf-8')
    return 2 if report['status'] in ('wsl_missing', 'wsl_error') else 0


if __name__ == '__main__':
    raise SystemExit(main())
