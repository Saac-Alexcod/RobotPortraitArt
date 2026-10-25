# Windows 一键安装器：行为、验收和限制

> **状态：功能分支候选版。** 安装脚本已在维护者电脑上通过已有环境复用、真实 APDrawingGAN 推理以及双端服务启动测试；**组员全新 Windows 电脑的首次安装仍未验证**。请勿宣称“任何电脑 100% 安装成功”。

**组员只想立即安装？** 先阅读 [《组员一页上手指南》](TEAMMATE_QUICKSTART.md)，从正确 Git 分支克隆，不要以默认 `main` 克隆后直接执行安装器。

## 1. 从功能分支克隆并运行

使用 Windows PowerShell：

```powershell
git clone https://github.com/Saac-Alexcod/RobotPortraitArt.git
cd RobotPortraitArt
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
```

克隆代码必须有 Git 和网络；安装器**不会替代 Git 的安装**。首次运行可能需要管理员授权、Ubuntu 首次账户初始化或 Windows 重启。遇到这些系统动作，处理后在项目根目录重复相同命令。默认情况下安装器不会自己重启 Windows 或升级显卡驱动。

## 2. 安装器真实执行的七个阶段

| 阶段 | 检查、选择、处理策略 |
|---|---|
| 1/7 Windows Python | 优先探测项目 `.venv`，再查 `py` 启动器和其他已安装 Python；无可用候选时尝试用 `winget` 安装 Python 3.11。 |
| 2/7 后端依赖 | 已存在 `.venv` 的解释器可运行且必要包能导入、`pip check` 通过时优先复用；新建环境优先 Python 3.10–3.12，安装 `requirements-web.txt`。 |
| 3/7 Node / 前端 | 兼容 Node 则复用，否则下载并验证项目隔离的 Node；依照锁文件运行 `npm ci`，复用已安装且匹配的依赖。 |
| 4/7 WSL | 探测已注册 WSL2 发行版和兼容性，优先复用可运行的 Ubuntu；必要时可尝试新增 Ubuntu-22.04。**`wsl.exe` 完全不可用时，目前仍会报错并指导人工启用 WSL，尚不能自动安装 Windows 可选功能。** |
| 5/7 Linux GAN 运行时 | 探测实际 WSL Python、关键推理模块、CUDA 计算能力；复用通过的环境，否则尝试在独立 venv 中安装兼容依赖。显式提供 `-APDPython` 时不会擅自修改那个解释器。 |
| 6/7 模型 | 检查 YuNet、PPHumanSeg、APDrawingGAN 官方权重，已验证的文件复用，缺少的尝试下载、校验并安装。 |
| 7/7 实际推理 | 采用内置示例真正执行 APDrawingGAN → Centerline SVG → 渲染图片，检查 Manifest/结果文件；未使用 `-NoStart` 时继续启动 FastAPI 与 Vite 并检查可访问性。 |

**版本兼容性**：Windows 后端和 WSL GAN 是两个不同的 Python 环境。已验证可工作的 Windows `.venv`（例如 Python 3.14）不因为 Linux PyTorch 的版本约束而被拒绝；新创建的 Windows 环境优先 3.11/3.12/3.10。固定的 WSL `torch==2.5.1`、`torchvision==0.20.1` 需使用兼容的 Linux Python（通常 3.9–3.12），并且必须通过实际 CUDA 和 GAN 推理测试。Ubuntu 24.04 并非一概不支持：只有**实际探测兼容**才可复用。缺少 GPU、驱动不兼容或组织网络策略禁止安装时，软件不能保证自动解决。

## 3. 命令选项（当前 `install.ps1` 的实际参数）

```powershell
# 默认：尽可能配置完整 GAN、真实推理、启动网页
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1

# 完成全部安装和实际出图但不启动前后端
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -NoStart

# 只读查看 WSL 候选；需要存在可用 Windows Python
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -CheckWSLOnly

# 只读环境诊断；需要可运行的 Windows Python
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -Doctor

# 仅 Baseline：跳过 WSL/GAN，不能算 GAN 成功
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -BaselineOnly

# 仅在已经确认现有 WSL Python 可运行时使用自定义解释器
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -Distro Ubuntu-22.04 -APDPython '/home/YOUR_USER/.venvs/robotportrait-apd/bin/python'

# 可选：可信代理，端口应替换为实际使用的端口
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -Proxy 'http://127.0.0.1:7890'
```

`-Proxy` 是当前安装进程临时设置，不会修改操作系统全局代理。WSL NAT 的 localhost 代理可达性取决于网络模式、防火墙和代理监听地址。切勿在命令行写明文代理密码、禁用 TLS 证书验证或提交个人配置。

## 4. 成功判定与人工网页验证

以下检查是**不同层次**：

1. **安装器端到端推理成功**：输出 `[PASS] First-run acceptance completed.`，结果在 `outputs/first_run_validation/`，验证 `manifest.json`、PNG、SVG。
2. **服务启动成功**：不带 `-NoStart` 运行后，确认 `http://127.0.0.1:8000/api/health` 与 `http://127.0.0.1:5173` 可访问；启动的两个服务窗口应保持运行。
3. **浏览器实际体验成功**：手动打开网页，选内置正面样例，选择 APDrawingGAN + Centerline，提交任务并下载结果。
4. **新电脑部署验证**：在不同于开发机的 Windows/WSL/GPU 环境从功能分支全新克隆、安装、生成，并再次执行确认可复用。

`GET /api/methods` 返回 HTTP 200 只说明接口可访问，应进一步确认 APDrawingGAN 标记 `available`，最后以真实生成结果为准。**Baseline 成功、CUDA 可用、单纯导入 torch 均不能替代完整 GAN 验收。**

> 网页地址需要用户手动在浏览器打开；脚本只是启动或复用服务进程，不保证自动弹出浏览器。API 和前端默认仅监听 `127.0.0.1`，不应直接对公网开放。

## 5. 需要人工参与、尚不保证自动恢复的情况

- **`wsl.exe` 不存在**、操作系统可选功能未启用、BIOS 虚拟化关闭或组织安全策略阻拦：先按 [微软 WSL 官方安装说明](https://learn.microsoft.com/windows/wsl/install) 处理，重启后复用安装脚本。
- NVIDIA GPU 不存在、Windows 显卡驱动不兼容、显存不足：先解决硬件与驱动要求，无法通过只下载 Python 包补齐。
- 第三方镜像/模型源 403、404、限流、受网络策略限制或需要许可：使用官方来源和授权访问方式，不关闭 HTTPS 验证。
- Windows 项目盘和 WSL 虚拟磁盘所在宿主盘空间不足：清理后再运行，不能只依据 WSL 中的 `df` 判断 Windows 剩余空间。
- 既有全局软件与模型依赖冲突：尽量使用项目独立虚拟环境；安装器不会强制覆盖组员全局 Python、Node、其他 Ubuntu 发行版或显式选定的 GAN 解释器。

命令卡住、安装失败和版本冲突：见 [首次安装故障排查手册](INSTALL_TROUBLESHOOTING.md)。日志位于 `outputs/first_run_setup.log`、`outputs/installer_logs/` 和 `outputs/first_run_validation/`。

## 6. 发布前应完成的验收

- 从 GitHub **功能分支**重新克隆（不能拷贝作者 `.venv`、`node_modules` 或模型权重）；
- Windows Python、Node、WSL 与 GPU/驱动跨机兼容性；
- 首次官方模型下载以及中断/重试；
- 真实 GAN → SVG → 浏览器提交/下载；
- 第二次运行不会重复下载已验证组件；
- 所有改动提交后在 GitHub 页面打开 README 与所有相对文档链接；
- 完整首次部署在组员机器通过后，再审议合并 `main`。

**维护提示**：本指南描述的是代码当前能力，不是保证任意陌生电脑一次执行就成功的承诺。
