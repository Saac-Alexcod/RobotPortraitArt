# 首次安装故障排查手册（Windows 本地部署）

> 适用于尚未合并 `main` 的首次安装候选版。此文档的目标是**失败不冒充成功、报告原因、保留证据、可重复运行**。第三方网络、GPU 驱动、系统权限等仍可能需要手动介入，不能承诺在任何设备上自动修复。

## 组员遇到失败时先看哪里

从项目**根目录**运行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
```

不要关掉窗口。记住失败摘要中的 **Stage、RPA 错误码、详细日志路径**，打开对应日志最后几行：

```powershell
Get-ChildItem .\outputs\installer_logs -Filter '*.log' | Sort-Object LastWriteTime -Descending | Select-Object -First 8 Name,LastWriteTime,Length
Get-Content .\outputs\first_run_setup.log -Tail 25
```

`outputs/installer_logs/` 记录每个外部安装命令的原始输出（stderr 合并入 stdout），`powershell_transcript.log` 尽可能记录 PowerShell 操作，`outputs/first_run_validation/*.log` 是最后的真实 GAN 出图结果。

**下载没有进度怎么办？** 新版下载器会报告已下载 MiB；若原生包管理器正在运行但不打印进度，监控器约每 15 秒打印一条仍在运行的提示。耗时阶段设有总超时，超时会停止相应子进程并保存日志，**不会显示成功**。Ctrl+C 取消后可重新执行安装命令。

## 错误码 → 操作建议

| 代码/阶段 | 可能原因 | 组员应采取的操作 |
|---|---|---|
| `RPA-E120` | 外部命令达到总超时 | 查对应的 `installer_logs/*.log` 末尾；检查磁盘、网络、代理、WSL 启动；修复后运行同一安装命令 |
| `RPA-E121` | 外部命令非零退出 | 打开失败步骤日志，查看真正报错（如 `pip`, `npm`, `apt`, 权限错误）；根据其内容处理，不要只盯退出码 |
| `RPA-E210` | Windows Python / venv / pip | 运行 `py -0p`；确认 Python 3.11、磁盘空间与 `requirements-web.txt` 可读取。若无法访问 PyPI，配置当前网络代理 |
| `RPA-E230` | Node.js / npm | `node -v`、`npm -v`、`npm ping`；检查 Node 版本、npm registry、代理与项目 `frontend/package-lock.json` |
| `RPA-E240` | WSL / Ubuntu | `wsl --status`、`wsl -l -v`；开启 BIOS 虚拟化，允许 UAC、根据提示重启 Windows，并至少启动 Ubuntu 一次创建 Linux 用户 |
| `RPA-E250` | WSL Python / PyTorch / CUDA | `wsl -d Ubuntu-22.04 -- nvidia-smi`；检查 NVIDIA Windows 驱动和 WSL2、PyTorch 安装日志以及下载源。没有支持的 GPU 无法通过下载补出本地 CUDA 推理 |
| `RPA-E260` | ONNX / GAN 模型下载或校验 | 检查 HTTP 状态码、TLS、代理、剩余磁盘空间、官方模型站点。`*.part` 会尽可能复用；ONNX SHA-256 不符不得使用。不要禁用证书验证 |
| `RPA-E270` | 实际生成 / 启动失败 | 查看 `outputs/first_run_validation/*.log` 与 `installer_logs/`；即使 CUDA=True，没有真实 SVG/Manifest 也不能报成功 |

### 网络/代理专项

使用可信的**本机代理**（根据你自己代理工具的实际端口填写）：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -Proxy "http://127.0.0.1:7890"
```

- `timed out` / `ConnectTimeout`：网络不可达、代理未启动或访问速度过慢；先检查代理是否监听，再运行相同命令。
- `HTTP 403`：模型源拒绝当前请求或需要许可/认证；检查官方页面。不要绕过访问控制。
- `HTTP 404`：上游文件地址变更；核对仓库内模型清单/官方文档，不要把 HTML 保存成权重。
- `HTTP 429`：限流；稍后重试，避免不断并发请求。
- `CERTIFICATE_VERIFY_FAILED`：系统根证书或公司代理证书问题；采用受信任证书或更换网络，**禁止**使用 `verify=False`。
- `WSL localhost proxy`：默认 NAT 网络下 `127.0.0.1` 可能指向 WSL 自身；安装向导尝试使用 Windows 网关，但防火墙/监听地址可能阻挡。仅在信任的本机网络下调整代理监听。
- `No space left on device` / `WinError 112`：先清出足够空间，避免重复下载模型，不要删除原有有效权重。

网络中断后的 `.part` 是**暂存下载**，不是可以直接用来推理的模型。不要手动重命名为 `.pt` 或 `.onnx`。完成后安装器会校验后再放入最终位置。

### 安装一半 Windows 重启了怎么办？

1. 正常完成系统提示的重启或 Ubuntu 首次用户创建。
2. 回到项目根目录，再执行同一条 `install.ps1` 命令。
3. 已经验证有效的虚拟环境、npm 依赖、模型文件会尽可能跳过或复用；仍损坏的文件会明确报告。
4. 只有真实 GAN 推理检查通过、服务成功启动，才视为整套部署完成。

## 如何安全地把错误发给项目维护者

发送失败阶段、错误码、**最后约 50 行**日志和电脑的 GPU 名称 / WSL 发行版状态即可。发出前检查日志里有没有 Windows 用户名、主目录、下载链接中的 token、代理账号或其他个人信息。不要发送密码、密钥或整个 `.venv`。部分安装程序输出可能包含第三方命令打印的敏感内容；虽然新监控器会尝试脱敏，**仍需人工复核**。

## 安全验证“错误提示能工作”——不用卸载 WSL

在项目根目录，Windows Python 环境可用时，执行下列命令模拟一个短时间挂住的子进程：

```powershell
& .\.venv\Scripts\python.exe .\scripts\installer_diagnostics.py --step "Simulated download stall" --timeout 2 --log .\outputs\installer_logs\simulated_timeout.log -- .\.venv\Scripts\python.exe -c "import time; time.sleep(30)"
```

**预期：** 大约两秒后输出 `RPA-E120`，命令退出码为 `124`，并在对应的 `.log` 保留诊断信息。这个模拟只启动一个睡眠 Python 进程，不会卸载软件、不下载模型。

## 版本兼容与自动选择（新增候选版）

默认不再强制要求 Ubuntu-22.04。安装器枚举 `wsl.exe --list --quiet`，逐一验证 WSL2 内核、Ubuntu/Debian、Python 版本及已有 CUDA 虚拟环境，优先复用**实际可用**的发行版（如 Ubuntu-24.04）。不能把“启动失败”解释为“未安装”。诊断 JSON 保存为 `outputs/installer_logs/wsl_discovery.json`。

当前固定推理依赖组合：`torch==2.5.1` / `torchvision==0.20.1` / CUDA 11.8 wheels。按照上游 torchvision 版本表，配套 Python 需要在 3.9–3.12 之内；Windows 项目环境采用 Python 3.10–3.12。Python 3.13/3.14、WSL1 或其他不支持此组合的环境不会直接被修改。优先使用兼容解释器创建专用 venv；无法安全适配时，安装器尝试**并行**部署 Ubuntu-22.04（需 UAC、支持的 GPU/Windows 驱动及可能的重启），保留原发行版。

| 新错误码 | 原因及建议 |
|---|---|
| `RPA-C101` | Windows `.venv` Python 不在已验证范围。先在项目外备份现有 `.venv`，安装器会利用另一个兼容 Windows Python 重建；不能直接覆盖其他项目的全局解释器。 |
| `RPA-C102` | WSL 缺少 Python 3.9–3.12、venv 包或现有 venv 不兼容。查看 `install_wsl_cuda_pytorch_packages.log`，确认 `python3 --version`、`apt-cache policy python3-venv`，必要时改用兼容 Ubuntu。 |
| `RPA-C103` | Node 版本低于 22.12。自动下载经 Node 官方 SHA-256 校验的 Windows Node 22 LTS 到 `.rpa_tools/`，不会覆盖全局 Node。 |
| `RPA-C110` | 已有 WSL 发行版存在但不兼容，或 WSL 探测失败。查看 `wsl_discovery.json`、`wsl --list --verbose` 和发行版内 Python；不可直接重装已有发行版。 |
| `RPA-E280` | 项目所在驱动器空间不足。先检查 Windows `Get-PSDrive` 和项目所在盘；至少预留数 GiB 用于 `pip`、Node、缓存与结果。 |
| `RPA-E281` | WSL 用户目录可用空间不足 8 GiB（或容量探测失败）。执行 `wsl -d <已选发行版> -- df -h ~`，确保 ext4 VHDX 和 Windows 宿主盘都有足够空间。 |
| `RPA-E282` | 新装 WSL 前 Windows 系统盘少于 10 GiB。清理系统盘或调整 WSL 存储位置后继续。 |

**用户应当先运行只读诊断：**

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -CheckWSLOnly
```

该命令不会安装任何系统组件、Python 包或下载模型。需要完整首次安装时再执行 `install.ps1`。Windows 驱动、BIOS 虚拟化、没有 NVIDIA GPU、公司防火墙或管理员权限被禁用均不能靠脚本强行修复；最终仍需真实 APDrawingGAN 生成成功才算通过。

### Windows Python 3.14 exists alongside Python 3.11/3.12

The Windows backend and the WSL APDrawingGAN runtime are separate environments.
A **working existing** project `.venv` using Python 3.14 is not automatically
incompatible with the web backend. The first-run installer now tries to reuse
such a venv after import and `pip check` validation. For a **new** Windows
`.venv`, it prefers Python 3.11/3.12 (or 3.10) via the Windows `py` launcher.
The pinned **Linux** PyTorch 2.5.1 runtime still requires its own supported
Python version; this policy has not been relaxed.

If the first stage nevertheless reports `No Python` or tries to install an
already-installed version, run the following from the repository root:

```powershell
& .\.venv\Scripts\python.exe -c 'import sys; print(sys.version); print(sys.executable)'
py -0p
py -3.11 -c 'import sys; print(sys.executable)'
winget list --id Python.Python.3.11 --exact
```

An executable listed by `winget` is not necessarily discoverable by `PATH`,
so the installer also checks the launcher and common installation locations.
Do not remove a functioning Python 3.14 venv just to satisfy the Linux GAN
Python requirement. If an existing `.venv` is actually broken, back it up
and repair in a separate environment before replacing it.

### APDrawingGAN WSL probe says `No module named visdom`

`visdom` is for training/visualization, whereas this project runs the upstream
`test.py` with `display_id = -1`. The first-run inference probe must **not**
require `visdom`; use the updated installer and check
`outputs/installer_logs/probe_wsl_pytorch_cuda.log`.
The inference-critical modules are `torch`, `torchvision`, `dominate`, `numpy`
and `PIL`. The updated probe also runs a small CUDA tensor operation.
Do not modify a functioning custom WSL interpreter simply to silence a
training-only import warning. Final success still requires real GAN output.
