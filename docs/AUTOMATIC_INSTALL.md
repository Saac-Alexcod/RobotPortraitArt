# Windows fresh-clone installer / GAN acceptance gate

> **Status: candidate, not certified on a fresh Windows computer.** Offline tests cannot prove that third-party model hosts, administrator elevation, NVIDIA drivers or WSL distribution provisioning will work on every PC. **Do not merge this installer into `main` until a teammate completes a real clean-machine test.**

## One entry point

On Windows 10/11 with Windows Package Manager (winget), from the **project root**, in ordinary PowerShell:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
```

The script uses a state-driven, repeatable process: Windows Python 3.11 / `.venv` / requirements → Node.js LTS / `npm ci` → WSL2 Ubuntu-22.04 → WSL Python venv / GPU-enabled pinned PyTorch → three official model files → preflight → **real APDrawingGAN + Centerline output** → launch FastAPI and Vite.

Each rerun checks existing prerequisites and **reuses** them. If WSL installation requires elevation, Windows reboot or Ubuntu first-launch user setup, perform that system operation, then **rerun the same command**. The script does not silently reboot, change device drivers, or modify global proxy settings. It does not modify an explicitly selected custom WSL Python.

Other useful entry points:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -Doctor
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -NoStart
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -Proxy 'http://127.0.0.1:7890'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -BaselineOnly
```

`-Doctor` is read-only, and requires an existing compatible Python to run its Python-based detector. `-NoStart` still performs a real GAN pipeline acceptance test but does not launch web processes. `-BaselineOnly` installs Windows/frontend dependencies and runs the basic web system without claiming GAN availability.

**Proxy support:** The proxy value is applied to **this installer process and subprocesses only**; it is not persisted in Windows settings. For WSL mirrored networking the localhost proxy can be used directly; for WSL NAT the script attempts to map a localhost Windows proxy to the current gateway. Firewalls, authenticated proxies, certificate trust and enterprise policies may require user intervention. Never place credentials in a command line. Downloads require valid TLS; no certificate bypass is used.


### 安装失败的诊断日志与恢复

耗时子进程有独立日志与总超时保护，模型下载会报告 MiB 进度；终端会提示错误码、故障阶段和恢复操作。
**请先阅读 [INSTALL_TROUBLESHOOTING.md](INSTALL_TROUBLESHOOTING.md)**，日志位于 `outputs/installer_logs/`，实际推理日志位于 `outputs/first_run_validation/`。首次部署不保证任意硬件及网络下 100% 成功；必须真实 GAN 出图才判定为成功。

## Non-negotiable acceptance requirements

1. Supported Windows host with installation permission; `winget` available and third-party package sources reachable.
2. CPU and memory sufficient for web frontend; for local GAN, a compatible **NVIDIA GPU and driver usable from WSL2**. Software cannot download a missing GPU.
3. WSL and Ubuntu user creation may require admin approval or reboot. Rerunning the installer resumes by rechecking actual state.
4. Model hosts and PyPI/PyTorch registries must be reachable through the current network/proxy.
5. The final command runs a **real** portrait through APDrawingGAN → vector SVG. It validates images, SVG and a completed manifest in `outputs/first_run_validation`. It **exits with an error** if all three frontal example candidates fail. Only a passed end-to-end run is considered successful.

## Frozen WSL runtime defaults

- Ubuntu-22.04 system Python 3.x and an **isolated** WSL venv at `$HOME/.venvs/robotportrait-apd`.
- PyTorch `torch==2.5.1` and `torchvision==0.20.1`, CUDA 11.8 wheels, pinned from [official PyTorch previous releases](https://pytorch.org/get-started/previous-versions/).
- Legacy compatibility packages installed only inside this dedicated WSL environment.
- Any already working WSL Python with torch, torchvision, dominate, numpy, PIL and CUDA is reused. A custom `-APDPython` path must be preconfigured; the installer will not alter it.

The selected wheel combination is documented upstream, **not yet validated on your teammate's GPU/driver**. If the driver cannot run CUDA 11.8 wheels, that requires an alternate supported combination and a new end-to-end test; it must not be marked as successful based solely on a version number.

## How to actually test it

### A. Existing configured development machine

Run `-Doctor`, then `-NoStart`. Expect it to skip installation and generate a real artifact. This is a regression test, **not** a clean-machine proof.

### B. Fresh checkout, same computer

Clone into a new folder without copying `.venv`, `node_modules`, `.onnx` or `.pt`. Run `install.ps1 -NoStart`. This exercises download and Windows venv setup, but still **shares WSL and GPU drivers** with the original machine.

### C. Teammate's clean Windows host (release gate)

Clone from the feature branch, run the single entry-point command; record whether any reboot/UAC prompt was needed and rerun after reboot if prompted. Save `outputs/first_run_setup.log`, `outputs/first_run_validation/*.log` (inspect/remove paths/personal data before sharing), and screenshot or inspect successful generated outputs.

Acceptance only if: (i) Python/backend/Node/Vite install, (ii) WSL/PyTorch CUDA detected, (iii) models validated, (iv) **real** inference passed, (v) `/api/health` and UI are accessible, (vi) second install invocation does not reinstall valid components, and (vii) offline or blocked network yields a clear non-success message and recovers on rerun.

Do not count Baseline success as GAN success. This is a project installer, not a driver/virtualization or network-administration bypass.

## Multiple Python and Ubuntu versions (compatibility candidate)

The first-run installer now performs a **non-destructive compatibility selection** before WSL installation. It probes registered WSL2 distros (including Ubuntu-24.04), looks for a supported Python or an already CUDA-ready project environment, and writes its decision to `outputs/installer_logs/wsl_discovery.json`. A distro that cannot start is never treated as absent. If a compatible distro exists, the installer must not call `wsl --install` for another distro.

For the frozen PyTorch `2.5.1` / torchvision `0.20.1` combination, Linux Python 3.9–3.12 is required by the published compatibility matrix; the Windows backend installer selects a discovered Python 3.10–3.12. Incompatible Python environments are not upgraded/downgraded globally. A dedicated Linux virtualenv is used, with a separate version-suffixed path when the previous project virtualenv uses an incompatible Python.

A compatible Node >=22.12 is reused; otherwise the installer downloads the verified official Node 22 LTS ZIP into ignored `.rpa_tools/`, preserving the user's system Node. Before large downloads the installer checks free disk space, but exact needs vary with cached packages, WSL VHD location and drivers. `pip check` and a real CUDA inference gate provide final validation. **This is not certified across all Windows/Ubuntu/GPU combinations**; release requires a teammate's fresh-machine trial.

More instructions and error codes: [INSTALL_TROUBLESHOOTING.md](INSTALL_TROUBLESHOOTING.md).


### Python version detection follow-up

Existing Windows `.venv` Python 3.10–3.14 is reused after import and `pip check`
validation, including Python 3.14 on machines where the backend already works.
**New** Windows virtual environments prefer a detected Python 3.11/3.12/3.10.
The separately installed WSL PyTorch environment retains its own Python 3.9–3.12
compatibility policy. Detection also examines the Windows `py` launcher,
including vendor-tagged `py -0p` entries.

The installer never guarantees that any Python 3.14 environment can install
every dependency; validation or package installation may still fail. The
intended behavior is to preserve working environments rather than refusing
them solely because of the interpreter's minor version.
