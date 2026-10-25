#!/bin/sh
# Called through wsl.exe -d SELECTED_DISTRO -- sh -s
# Never changes the global default Python or removes an existing venv.
set -eu
BASE="$HOME/.venvs/robotportrait-apd"

select_python() {
  for exe in python3.11 python3.10 python3.12 python3.9 python3; do
    if ! command -v "$exe" >/dev/null 2>&1; then continue; fi
    version=$($exe -c 'import sys;print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)
    case "$version" in 3.9|3.10|3.11|3.12) printf '%s\n' "$exe"; return 0;; esac
  done
  return 1
}

PYBASE=$(select_python) || {
  echo '[RPA-C102] No Python 3.9-3.12 found. The pinned PyTorch 2.5.1 runtime cannot be installed here.' >&2
  echo '[RPA-C102] Use an existing compatible WSL2 Ubuntu (22.04/24.04), or install a supported Python alongside system Python.' >&2
  exit 12
}
VERSION=$($PYBASE -c 'import sys;print("%d%d" % sys.version_info[:2])')
VENV="$BASE"
if [ -x "$BASE/bin/python" ]; then
  if ! "$BASE/bin/python" -c 'import sys;assert (3,9)<=sys.version_info[:2]<=(3,12)' >/dev/null 2>&1; then
    VENV="${BASE}-py${VERSION}"
    echo "[RPA-C102] Existing venv uses an incompatible Python; preserving it. Isolated fallback: $VENV"
  fi
fi

if [ ! -x "$VENV/bin/python" ]; then
  mkdir -p "$(dirname "$VENV")"
  echo "[RPA] Creating isolated venv with $PYBASE at $VENV"
  if ! "$PYBASE" -m venv "$VENV"; then
    echo "[RPA-C102] $PYBASE cannot create a virtualenv; install its matching venv module inside WSL." >&2
    exit 13
  fi
fi
PY="$VENV/bin/python"
if ! "$PY" -c 'import sys;assert (3,9)<=sys.version_info[:2]<=(3,12)' 2>/dev/null; then
  echo '[RPA-C102] Virtual environment Python is incompatible; refusing to install.' >&2
  exit 14
fi

# Training-only Visdom is not needed for this project's test.py inference.
if "$PY" -c 'import torch,torchvision,dominate,PIL,numpy; assert torch.cuda.is_available()' 2>/dev/null; then
  echo "RPA_PYTHON=$PY"
  exit 0
fi

if ! (command -v nvidia-smi >/dev/null 2>&1 || [ -x /usr/lib/wsl/lib/nvidia-smi ]); then
  echo '[RPA-E250] NVIDIA GPU is unavailable from WSL2. CUDA cannot be fixed by pip/weight downloads.' >&2
  exit 15
fi

# Versions documented in scripts/installer_compatibility.py and official PyTorch releases.
"$PY" -m pip install --disable-pip-version-check --retries 4 --timeout 45 --upgrade pip
"$PY" -m pip install --disable-pip-version-check --retries 4 --timeout 45 'numpy==1.26.4' 'Pillow==10.4.0' 'scipy==1.13.1' 'dominate>=2.3.1,<3' 'opencv-python-headless>=4.8,<5'
"$PY" -m pip install --disable-pip-version-check --retries 4 --timeout 90 torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
"$PY" -m pip check
"$PY" - <<'PY'
import sys, torch, torchvision, dominate, numpy, PIL
assert (3,9) <= sys.version_info[:2] <= (3,12)
print('torch:', torch.__version__, 'torchvision:', torchvision.__version__)
print('CUDA available:', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('[RPA-E250] CUDA not usable; verify GPU driver and WSL2 compatibility.')
print('GPU:', torch.cuda.get_device_name(0))
PY
echo "RPA_PYTHON=$PY"
