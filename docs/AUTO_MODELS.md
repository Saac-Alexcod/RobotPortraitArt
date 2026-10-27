# APDrawingGAN 与 GANs N’ Roses 自动下载、运行

在项目根目录的 PowerShell 执行：

```powershell
# 只下载模型，不安装环境、不启动服务；仅需 Python 3.10+
python scripts/download_models.py
# 也可以只下载一种模型
python scripts/download_models.py --model apdrawinggan
python scripts/download_models.py --model cartoon

# 首次：准备 Python 依赖、WSL CUDA 环境、下载模型并运行内置示例
.\scripts\run_models.ps1 -Setup

# 已有环境：运行两个模型
.\scripts\run_models.ps1 -InputImage "D:\photos\portrait.jpg"
# 只运行一个模型
.\scripts\run_models.ps1 -Model apdrawinggan -InputImage "D:\photos\portrait.jpg"
.\scripts\run_models.ps1 -Model cartoon -InputImage "D:\photos\portrait.jpg"
```

运行前需已安装 WSL2 Ubuntu、Linux Python 3.10–3.12（含 venv）及支持 CUDA 的 NVIDIA 驱动。`-Setup` 会下载较大的 PyTorch 包；已有可用环境时省略该选项。支持 `-Distro` 和 `-APDPython` 指定已有环境；`-Setup` 与 `-APDPython` 不能同时使用。

未指定图片时使用内置正面合成人像。默认输出到 `outputs/models/apdrawinggan/` 和 `outputs/models/cartoon/`，包含风格图、SVG 和运行记录。用 `-OutputDir` 修改输出位置，`-Seed` 设置随机种子（默认 42）。模型分别使用 512 和 256 分辨率。一个模型失败仍会尝试另一个，最终返回非零状态。运行入口直接处理图片，不启动 Web 服务。

已有环境也可直接执行：

```powershell
.\.venv\Scripts\python.exe scripts/run_models.py --input "D:\photos\portrait.jpg"
```

权重按项目固定 SHA256 校验，损坏文件自动重新下载，下载完成并校验后才替换目标文件，失败最多尝试三次；有效本地文件直接复用。APDrawingGAN 优先复用根目录已有的官方 `APDrawingGAN-Models1.zip`，仅提取指定 checkpoint。APDrawingGAN 源码使用项目已包含的适配版本。GANs N’ Roses 源码沿用固定上游版本。哈希来自本项目已使用的模型文件，不是上游数字签名；模型目录的 `provenance.json` 记录来源及哈希。不要同时启动多个下载进程。

下载需能访问清华模型站、GitHub 和 Google Drive；Google Drive 配额、登录页或内容校验失败会报错，修复网络后重试。下载脚本不要求 WSL/GPU。普通运行不自动安装依赖或下载模型；需要时使用 `-Setup`。已有自定义 WSL 环境的用户先单独下载，再通过 `-APDPython` 运行。

Web 用法：先执行下载脚本，然后运行 `.\scripts\start_web.ps1`。原有 `start_web.ps1 -SetupAPD` 仍只负责安装 WSL 环境，模型下载使用上述入口。
