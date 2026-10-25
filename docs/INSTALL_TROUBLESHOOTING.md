# 组员安装失败：错误码、日志和恢复步骤

> 本手册适用于默认 `main` 分支的 Windows 本地安装器。出现问题时，优先查看详细日志，不要贸然删除已有 Python、WSL、虚拟环境或模型。

## 1. 安装失败的第一反应

从项目根目录执行以下**只读操作**：

```powershell
Get-Content .\outputs\first_run_setup.log -Tail 40
Get-ChildItem .\outputs\installer_logs -File |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 10 Name,LastWriteTime,Length
```

记下终端的 `[FAILED]`、`Stage:`、错误码（`RPA-...`）及对应日志文件。查看相关日志最后 80 行：

```powershell
Get-Content '.\outputs\installer_logs\步骤名.log' -Tail 80
```

请把 `步骤名.log` 换成终端**实际显示的文件名**。安装阶段日志在 `outputs/installer_logs/`；总阶段日志在 `outputs/first_run_setup.log`；GAN 真实结果与日志在 `outputs/first_run_validation/`。

## 2. 常见错误及对应操作

| 代码/症状 | 可能原因 | 先做什么（可执行） |
|---|---|---|
| 无法执行 `git clone` / `git` 不存在 | Git 未安装、网络或认证问题 | 安装 Git for Windows 后重开 PowerShell，`git --version`；确认从默认 `main` 分支克隆仓库。 |
| 运行 `install.ps1` 提示文件不存在 | 克隆的是 `main`、目录不对 | `git branch --show-current`、`Test-Path .\install.ps1`；重新克隆**功能分支**。 |
| `RPA-E210`、找不到 Python/venv | Python 未安装或候选不兼容、`pip` 报错 | `py -0p`、`Test-Path .\.venv\Scripts\python.exe`；查看 `install_windows_python_requirements.log`。已有可工作的 `.venv` 不要删除。 |
| `RPA-C101` | 已有项目 `.venv` 的 Python 不可用或版本不在支持范围 | 保留现场并备份 `.venv` 到项目外，定位具体解释器；**不要卸载全局 Python**。使用兼容的新解释器单独重建。 |
| `RPA-E230` / `RPA-C103` | Node 版本不适合 Vite、npm 下载失败 | `node --version`、`npm --version`；查看 `install_isolated_compatible_node_lts.log` / `install_frontend_npm_dependencies.log`。项目会优先用独立 Node，不应随意升级系统全局 Node。 |
| `RPA-E240` / `wsl.exe unavailable` | Windows WSL 可选功能/启动器不存在 | 当前安装器**不会自动启用这个缺失的系统组件**。参考[微软 WSL 安装文档](https://learn.microsoft.com/windows/wsl/install)，按提示申请管理员权限、启用并重启后再运行。 |
| `RPA-E242` / 已注册 Ubuntu 但无法启动 | WSL2 发行版启动异常 | `wsl --list --verbose`；查看 `probe_wsl_readiness.log`。**不要把已存在的发行版直接重装**。 |
| `RPA-C110` / 其他 Ubuntu 已存在但不兼容 | WSL1、Python/虚拟环境不兼容或检测出错 | `wsl --list --verbose`、查看 `wsl_discovery.json`，检查所选发行版里 Python。安装器应优先复用经过验证的 Ubuntu 24.04 等版本，而不是强行覆盖。 |
| `RPA-E243` / 新 Ubuntu 安装后未就绪 | 系统重启、首次用户初始化尚未完成 | 按 Windows/Ubuntu 提示处理、手动启动一次 Ubuntu 初始化账户，然后**重新运行同一 `install.ps1`**。 |
| `RPA-E250` / `CUDA False` / PyTorch 安装失败 | WSL CUDA、驱动、Linux Python、网络或包冲突 | 在 WSL 内检查 GPU (`wsl -d <DISTRO> -- nvidia-smi`，请将占位符替换为实际名字)、`probe_wsl_pytorch_cuda.log`，不要直接安装 Windows torch 来代替 WSL 环境。 |
| `No module named visdom` | 老版安装器错误将训练可视化包当成推理依赖 | 从最新功能分支拉取安装器；新版推理检查只要求真正必要模块和 CUDA。不要给已经能推理的环境强装 Visdom。 |
| `RPA-E260` / 权重校验失败 | 权重下载不完整、上游 403/404、目录或许可问题 | 查 `download_verify_official_model_files.log`，检查官方来源。保留有效权重；不使用校验失败的下载文件。 |
| `RPA-E270` / 真实 GAN 生成失败 | 推理进程、模型、输入或 SVG 输出失败 | 查 `outputs/first_run_validation/*.log`；即便 CUDA=True，缺少有效 SVG/Manifest 也不能判为安装成功。 |
| `RPA-E120` | 超时，子进程长时间不退出 | 检查错误步骤日志、磁盘、网络、代理；解决后重新运行，已验证文件尽量复用。 |
| `RPA-E121` | 安装子进程以非零码退出 | 查看对应 `.log` 中最后的 `pip`、`npm`、`apt` 或 `winget` 原始报错，再针对实际原因处理。 |
| `RPA-E280 / E281 / E282`、`No space left`、`WinError 112` | 项目盘、WSL 内部或 Windows 系统盘容量不足 | 检查 `Get-PSDrive -PSProvider FileSystem` 与 `wsl -d <DISTRO> -- df -h ~`（替换实际发行版）；还要确认宿主机存放 WSL VHDX 的盘有空间。 |

**命令中的 `<DISTRO>` 是说明用的占位符，不能原样复制。** 例如当前安装器选择 Ubuntu-22.04 时，用 `wsl -d Ubuntu-22.04 -- nvidia-smi`。

## 3. 网络下载卡住、代理和校验问题

- **连续等待但没有输出**：监控器会输出心跳，外部阶段设有总超时；超时会留下 `RPA-E120` 和阶段日志。安装器**不能保证**任何外部下载源都支持精确断点续传。
- **`HTTP 403/404/429`**：分别检查权限/页面已变更/限流；优先使用官方上游页面，不关闭权限控制，不把 HTML 下载页面当权重。
- **`CERTIFICATE_VERIFY_FAILED`**：检查系统证书和可信代理；不要为了下载成功而禁用 HTTPS 证书验证。
- **`Proxy 407`**：需要代理身份验证，咨询网络管理员；不要将代理密码保存在仓库或终端历史。
- **`WSL localhost proxy` 警告**：如果 CUDA 检测、`pip` 与 GAN 推理最终成功，可忽略警告。**如果网络下载失败**，Windows 的 `127.0.0.1` 在 WSL NAT 中可能不可直接访问，需检查代理监听地址、网关、防火墙或 WSL 镜像网络模式。
- **`.part` 下载残片**：它不是 `.pt`、`.onnx`；不要手动改名。安装器在可行时重试/复用，服务器不支持 Range 时可能重新开始下载。

使用可信代理时，可在项目根目录执行（替换真实端口）：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -Proxy 'http://127.0.0.1:7890'
```

`-Proxy` 不修改 Windows 全局代理。若涉及公司代理证书或防火墙，应寻求授权，不应绕过安全策略。

## 4. 安装中途断电、关闭终端或重启

1. 先完成 Windows 的 UAC/重启或 Ubuntu 首次用户初始化，不要随意删除已经下载完成的有效模型。
2. 再次进入项目目录，**运行之前那条完全相同的 `install.ps1` 命令**（包括此前使用的 `-Proxy` 等参数）。
3. 安装器会重新检测当前状态、复用已验证的 `.venv`、Node/npm、模型和可用 WSL 运行环境；但某些包管理器和模型服务器不能保证从中断位置无损续传。
4. 只有看到 `[PASS] First-run acceptance completed.` 并完成网页 APDrawingGAN 生成，才宣布组员部署成功。

## 5. 常见的误解

**Q：Windows Python 3.14 已有且可运行，为什么 GAN 要检查 Linux Python 3.10？**

A：Windows Web 后端与 WSL 中的 APDrawingGAN 是**两套独立虚拟环境**。现有 Windows `.venv` 若依赖测试通过，可复用 3.14；GAN 的固定 PyTorch 组合要求另一个兼容版本。不要删除已通过测试的环境来满足另一边的要求。

**Q：为什么显示 `Ubuntu-22.04 Stopped`？**

A：`Stopped` 是当前未运行，不表示没安装。是否可用以发行版启动测试、Python/CUDA 检查为准。

**Q：`pytest` 提示 `LPIPS 不可用：No module named torch`？**

A：这是 Windows 测试环境里的**可选 LPIPS 评价指标**警告，不等于 WSL GAN 推理不可用。仍以实际 GAN 输出为准。

**Q：前端页面打开了，但 APDrawingGAN 方法不可用？**

A：可能只启动了 Baseline，或者启动的是旧的后端进程。先查看 `GET /api/methods?refresh=true` 和后端窗口；确认完成真实 GAN 生成后再判断成功。`-BaselineOnly` 不等于 GAN 可用。

## 6. 如何向项目维护者提交故障信息

建议发送：失败阶段、具体 RPA 错误码、最近的相关步骤日志**最后约 50–80 行**、Windows 版本、GPU 型号、`wsl --list --verbose` 输出。日志中可能包含真实用户名、磁盘目录、代理 URL、账号或访问令牌，**发送前先脱敏**。不要公开密码、token、完整 `.venv` 或私人照片。

可先进行**不安装任何环境**的检查：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -CheckWSLOnly
```

注意这个只读操作本身依赖一个可运行的 Windows Python；如果电脑连 Python 都没有，它可能提示缺少诊断解释器，此时应直接运行正常首次安装入口，而不是误认 WSL 坏了。

更多说明：[一页快速上手](TEAMMATE_QUICKSTART.md) · [自动安装阶段](AUTOMATIC_INSTALL.md) · [模型与权重](MODEL_SETUP.md)。
