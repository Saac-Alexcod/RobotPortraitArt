# First-run setup and network troubleshooting (Windows / WSL2)

**New:** The proposed single-entry installer is documented in [AUTOMATIC_INSTALL.md](AUTOMATIC_INSTALL.md). The older commands below remain available for manual diagnosis and selective downloads. Do not treat the installer as fresh-Windows verified until the teammate completes the acceptance test.

This guide targets a **fresh GitHub clone or downloaded ZIP**. It does not assume a specific Windows username, drive letter, WSL home directory, or pre-existing `.venv`. All commands are run from the **project root in PowerShell**.

## A. Modes and safety

- `scripts/setup_environment.ps1` without flags runs **read-only checks**. It does not install WSL, Python, PyTorch, CUDA, or download weights.
- `-InstallWindowsDeps` explicitly creates the Windows `.venv` if absent and installs `requirements-web.txt` (needs Python already installed).
- `-DownloadModels` explicitly downloads YuNet and PPHumanSeg from OpenCV Zoo, **checks their pinned SHA-256 hashes**, and atomically installs them.
- `-DownloadAPD` explicitly downloads the official upstream Model1 archive and extracts only the generator checkpoint files it recognizes. This archive does **not** have a SHA-256 pinned by this repository; you can optionally supply a previously verified hash via `-APDChecksum`. Upstream network availability and third-party distribution terms are not under our control.
- Startup (`scripts/start_web.ps1`) performs read-only environment checks, **never downloads/install packages**, and continues in Baseline mode when optional GAN setup is unavailable (unless you specify `-RequireAPDrawingGAN`).

## B. Normal installation

First install Python 3.11, Node.js and Git using trusted vendor installers. WSL and an NVIDIA GPU are not required for the OpenCV Baseline.

```powershell
# 1. Read-only diagnosis
.\scripts\setup_environment.ps1

# 2. Set up the Windows backend environment (opt-in)
.\scripts\setup_environment.ps1 -InstallWindowsDeps

# 3. Download the two official OpenCV ONNX models (opt-in)
.\scripts\setup_environment.ps1 -DownloadModels
```

Optional APDrawingGAN users:

```powershell
# 4. Download upstream Model1 weights (opt-in)
.\scripts\setup_environment.ps1 -DownloadAPD

# 5. Recheck model files, selected WSL distro, WSL Python, torch/torchvision, CUDA
.\scripts\setup_environment.ps1 -Distro Ubuntu-22.04

# 6. For GAN users, verify full legacy APDrawingGAN preflight
python scripts/check_apdrawinggan.py --distro Ubuntu-22.04 --python /home/YOUR_USER/.venvs/robotportrait-apd/bin/python
```

**Windows native PyTorch alone is not enough for this release**: the APDrawingGAN runner currently invokes `wsl.exe`. Inside WSL2, provision a dedicated Python environment and compatible PyTorch/torchvision; follow [PyTorch's official installer](https://pytorch.org/get-started/locally/) and the [upstream APDrawingGAN documentation](https://github.com/yiranran/APDrawingGAN). The older research repository originally targeted Python 2.7/PyTorch 0.4+, and the combination used by the project may need compatibility adjustments. The setup assistant therefore **does not blindly install a new PyTorch/CUDA build**.

Run the backend after diagnosis:

```powershell
.\scripts\start_web.ps1
```

The launcher automatically tries `~/.venvs/robotportrait-apd/bin/python` in the chosen WSL distribution (instead of `/home/winter/...`). If your environment lives elsewhere:

```powershell
.\scripts\start_web.ps1 -Distro Ubuntu-22.04 -APDPython /home/YOUR_USER/special-env/bin/python
```

## C. Proxy and blocked downloads

Windows and WSL may use **different network proxy settings**. For Windows-side model downloads you may pass a proxy **only for the current invocation**:

```powershell
.\scripts\setup_environment.ps1 -DownloadModels -Proxy 'http://127.0.0.1:7890'
.\scripts\setup_environment.ps1 -DownloadAPD -Proxy 'http://127.0.0.1:7890'
```

Without `-Proxy`, Python normally honors `HTTPS_PROXY` / `HTTP_PROXY` environment variables and configured Windows proxies. `-NoProxy` disables that detection for model downloads. **Avoid including proxy passwords in commands**, because shell history can retain them. Do not disable TLS verification to get around certificate errors; repair the trusted CA configuration instead.

For WSL `pip`, Windows-side `-Proxy` does *not* automatically change its environment. Check networking inside the selected distro. Windows 11 22H2+ supports `autoProxy=true` and `networkingMode=mirrored` in `%USERPROFILE%\.wslconfig`, but these are **system-wide choices** and must be applied manually after reviewing [Microsoft's WSL networking guide](https://learn.microsoft.com/windows/wsl/networking). Do not overwrite another user's `.wslconfig`. WSL might need `wsl --shutdown` before changes take effect.

| Symptom | Cause to inspect | Safe recovery |
|---|---|---|
| Missing WSL or distribution | WSL not installed, distro name differs, WSL1 | `wsl --list --verbose`, then use Microsoft's `wsl --install` instructions; reboot if prompted |
| Missing WSL Python | Wrong interpreter path, missing `venv` | Create a dedicated environment inside WSL; use `-APDPython` |
| PyTorch imports but CUDA is False | CPU-only wheel, incompatible driver, WSL GPU unavailable | Check `nvidia-smi` and official PyTorch selector; don't reinstall blindly |
| HTTP 403/404 or invalid ZIP | Upstream host blocked/moved/needs interactive access | Download from the official upstream page with a browser, then place weights in the expected directory |
| Proxy 407 | Authenticated proxy required | Ask the network administrator, avoid storing credentials in source/history |
| DNS/timeout | Proxy, DNS, firewall, VPN, remote endpoint | Verify Windows and WSL networking independently; retry with a trusted approved proxy |
| SHA-256 mismatch | Partial/corrupt download, HTML or LFS pointer, changed model | Download script deletes the bad partial file and does not overwrite an existing model |
| Missing weights after download | Incorrect ZIP layout or access problem | Downloader fails safely; inspect official archive and `docs/MODEL_SETUP.md` |

Partial downloads use `.part` files and attempt an HTTP Range resume on retry; if the server ignores Range the download starts over. The downloader never extracts arbitrary ZIP paths and never overwrites an existing checkpoint. On network errors, a partial archive may remain so a later invocation can resume.

## D. Diagnostics and offline use

```powershell
# Windows venv not yet set up? Use system Python 3.11 for read-only diagnostics:
py -3.11 scripts/doctor_environment.py --distro Ubuntu-22.04
py -3.11 scripts/bootstrap_models.py

# Export a report without uploading personal information:
py -3.11 scripts/doctor_environment.py --json outputs/setup_diagnostic.json
```

You may obtain model files manually and place them at the exact locations in `docs/MODEL_SETUP.md`; the doctor then works offline. This release does not promise every legacy APDrawingGAN dependency runs on every GPU/driver combination. Full readiness still requires an actual image inference, not just a green dependency check.

**Model URLs**: [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet), [PPHumanSeg](https://github.com/opencv/opencv_zoo/tree/main/models/human_segmentation_pphumanseg), [APDrawingGAN Model1](https://github.com/yiranran/APDrawingGAN). Only download weights whose usage and distribution terms you have reviewed.

### 多版本 Windows Python 与其他 Ubuntu 发行版

新版安装器先探测 Windows Python 3.10–3.12，再枚举 WSL2 发行版及其实际 Python/CUDA 能力。若 Ubuntu-24.04 可用，会直接复用，不会因为不是 Ubuntu-22.04 而重装。固定 PyTorch 2.5.1 / torchvision 0.20.1 组合要求 Python 3.9–3.12；如系统 Python 不适配，会在项目专用虚拟环境中尝试兼容解释器，而非替换全局 Python。Node.js 版本不兼容时使用项目内的官方哈希校验版。

磁盘不足、驱动不兼容或包版本冲突会以非零退出码终止，并显示阶段、详细日志路径和错误码，详见 [INSTALL_TROUBLESHOOTING.md](INSTALL_TROUBLESHOOTING.md)。以上改动仍需要组员在真实 Windows 电脑验证，**不能仅凭本机已有环境成功就声称所有电脑都会 100% 成功**。
