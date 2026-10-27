# RobotPortraitArt

RobotPortraitArt 将人像照片转换为艺术肖像和有序 SVG 绘制轨迹。项目提供网页工作台和命令行流水线，可选择不同的人像生成方式、矢量化方式，并在浏览器中预览 SVG 的逐笔绘制过程。

当前项目实现的是图像处理、二维路径规划和软件绘制模拟，不包含实体机械臂驱动、运动学或硬件闭环控制。

## 项目使用的技术

| 部分 | 技术与用途 |
| --- | --- |
| 后端 | Python、FastAPI、Uvicorn；接收图片、创建和查询任务、提供结果文件与 API 文档。 |
| 图像处理 | OpenCV、NumPy、Pillow、scikit-image、PyYAML；完成人像处理、线稿生成及配置读取。 |
| 前端 | Vue 3、TypeScript、Vite；上传图片、选择生成和矢量化方法、查看结果及绘制动画。 |
| 人像生成 | OpenCV Baseline 可直接运行；APDrawingGAN 和 GANs N' Roses 是需要额外模型及 WSL/CUDA 环境的可选方法。 |
| 矢量化与路径 | Contour 和 Centerline 将图像转换为 SVG 路径；Centerline 还包含骨架图整理和笔画路径规划。 |
| 输出 | 人像及中间结果图片、SVG、任务清单 JSON；网页提供结果预览和下载。 |

仓库提供合成示例图片，可用于体验，不需要准备 CelebA 数据集。模型权重、自动人脸检测和人体分割的 ONNX 文件没有随仓库分发；它们只在启用相应模型时需要。

## 环境要求

推荐使用 Windows 10/11 64 位和 PowerShell。完整 GAN 部署还需要 WSL2/Ubuntu、支持 WSL CUDA 的 NVIDIA 显卡及驱动，并需要下载模型文件。

| 使用方式 | 需要的软件 |
| --- | --- |
| 网页 + OpenCV Baseline | Python 3.10 及以上、Node.js 22.12 及以上（含 npm）。不要求 WSL、NVIDIA GPU 或 GAN 权重。 |
| APDrawingGAN / GANs N' Roses | 在 Baseline 基础上增加 WSL2/Ubuntu、可用的 NVIDIA WSL CUDA 环境和对应模型权重。 |
| 克隆仓库 | Git。若代码已经下载到本机，则不需要 Git。 |

Python 3.11 是手动新建虚拟环境的推荐版本。项目中的 `requirements-web.txt` 安装后端及图像处理依赖；`frontend/package-lock.json` 锁定前端依赖。`requirements-optional.txt` 中的 LPIPS 和 CairoSVG 用于可选评估或 SVG 回栅格化，不是网页启动的必需依赖。

## 方式一：使用安装脚本

在 Windows PowerShell 中进入仓库根目录运行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
```

默认安装流程会检查或准备 Windows Python 虚拟环境、后端依赖、Node/npm、WSL Ubuntu、WSL 中的 PyTorch/CUDA 运行环境和模型文件；随后实际运行 APDrawingGAN 与 GANs N' Roses 的生成验收，并启动后端和前端。完成后在浏览器打开 `http://127.0.0.1:5173`。服务终端会显示 API 地址，默认 API 文档地址为 `http://127.0.0.1:8000/docs`。

完整安装需要联网并会下载较大的软件包和模型。首次安装 WSL 可能需要管理员授权、完成 Ubuntu 初始化或重启 Windows；如果系统尚未启用 WSL/虚拟机平台组件，需按 Windows 提示启用后再重新运行安装命令。安装器会尽量复用已有环境，失败时可以修复对应问题后再次运行。

没有 NVIDIA GPU 或只想试用 Baseline 时，运行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -BaselineOnly
```

`-BaselineOnly` 会安装网页所需依赖并启动服务，不配置 WSL/CUDA/GAN 模型。若要完成完整环境和模型验收、但暂时不启动网页，可加 `-NoStart`。安装日志位于 `outputs/first_run_setup.log` 和 `outputs/installer_logs/`。

## 方式二：手动安装和运行

以下步骤不运行 `install.ps1`，以 Windows PowerShell 为例。命令从仓库根目录执行。

### 1. 创建 Python 环境并安装后端依赖

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-web.txt
```

如果电脑上的 Python 版本不是 3.11，请把 `py -3.11` 换成已安装的 Python 3.10 或更新版本。此处使用项目虚拟环境中的 Python，后续后端命令也使用同一路径。

### 2. 启动后端

在一个 PowerShell 窗口中从仓库根目录运行：

```powershell
.\.venv\Scripts\python.exe -m uvicorn robot_portrait.web.app:app --app-dir src --host 127.0.0.1 --port 8000
```

后端运行期间保持该窗口打开。健康检查地址为 `http://127.0.0.1:8000/api/health`，交互式 API 文档为 `http://127.0.0.1:8000/docs`。

### 3. 安装并启动前端

第一次使用时，在另一个 PowerShell 窗口执行 `npm ci` 安装锁定版本的前端依赖；以后启动时只需运行 `npm run dev`：

```powershell
Set-Location .\frontend
npm ci
npm run dev
```

保持前端窗口打开，在浏览器访问 `http://127.0.0.1:5173`。如果 Vite 输出了其他端口，以终端显示的地址为准。前端默认连接 `http://127.0.0.1:8000`；若后端改用其他地址，可在启动 Vite 前设置 `ROBOT_PORTRAIT_API_URL`。

### 4. 运行命令行流水线（可选）

网页以外，也可以直接对示例图片运行 Baseline + Centerline：

```powershell
.\.venv\Scripts\python.exe .\scripts\run_pipeline.py `
  --input .\data\examples\portrait_sample.jpg `
  --output-dir .\outputs\sample_baseline `
  --method baseline `
  --config .\configs\centerline.yaml `
  --size 512
```

结果写入 `outputs/sample_baseline/`。更多公开示例及适用说明见 [`data/examples/README.md`](data/examples/README.md)。

## 可选 GAN 模型

网页和命令行默认都可使用 OpenCV Baseline；启动后端时若 WSL/GAN 环境不可用，Baseline 仍可运行。网页会根据后端报告的模型状态启用或禁用对应选项。

- **APDrawingGAN**：需要 WSL2 中可进行 CUDA 推理的 Python/PyTorch 环境，以及 APDrawingGAN checkpoint、YuNet 人脸检测 ONNX 和 PPHumanSeg 人体分割 ONNX。仓库不包含这些权重。下载位置、环境检查和手动配置见 [`docs/MODEL_SETUP.md`](docs/MODEL_SETUP.md)。
- **GANs N' Roses**：需要 WSL/CUDA 推理环境及该模型权重。安装脚本会尝试准备并验收此模型；手动接入说明见 [`docs/CARTOON_SETUP.md`](docs/CARTOON_SETUP.md)。

模型依赖不等于模型权重：仅安装 Python 包或 WSL 环境不会自动让 GAN 可用。不要将第三方 checkpoint、ONNX 权重或 CelebA 人像数据提交到仓库；下载及使用时遵守各上游项目的许可条款。

### 自动下载模型须知（APDrawingGAN 与 GANs N’ Roses）

在项目根目录的 PowerShell 中执行，下载脚本仅需 **Windows Python 3.10+**，无需 WSL 或 GPU：

```powershell
# 下载两个模型，以及 APDrawingGAN 所需的 YuNet / PPHumanSeg 权重
python scripts/download_models.py

# 按需只下载一种模型
python scripts/download_models.py --model apdrawinggan
python scripts/download_models.py --model cartoon
```

- **网络要求：** 需要能访问清华模型站、GitHub（含原始文件下载域名）和 Google Drive。Google Drive 配额限制、登录页或内容校验失败都会报错；排除网络或下载源问题后重新执行命令即可。
- **下载位置与大小：** APDrawingGAN 权重安装到 `third_party/APDrawingGAN/checkpoints/formal_author/`，预处理权重安装到 `third_party/models/`；GANs N’ Roses 源码和约 302 MiB 的权重安装到 `third_party/GANsNRoses/`。APDrawingGAN 解压后的 checkpoint 约 243 MiB，另需为下载压缩包和临时文件预留空间。
- **校验与重复执行：** 权重按项目固定 SHA256 校验，校验通过的本地文件直接复用，损坏文件自动重新下载；只有下载完成并校验通过才替换目标文件。哈希来自本项目已使用的模型文件，不是上游数字签名；模型目录的 `provenance.json` 记录来源和哈希。
- **失败重试：** 每个文件下载失败最多尝试三次。中断后可重新运行；未完成的文件会重新下载，不支持断点续传。请勿同时启动多个下载进程。
- **源码自动补齐：** APDrawingGAN 自动检查 `third_party/APDrawingGAN/`，优先从本地 Git 历史恢复缺失文件；无法恢复时，从 `codeload.github.com` 下载项目固定提交 `13a03f7` 的兼容源码快照，按随脚本提供的清单校验每个下载文件的大小和 SHA256。无需手动克隆 APDrawingGAN，也不要求本机安装 Git。已有源码文件和模型权重不会被覆盖。`download_models.py`、`bootstrap_models.py --download-apd` 和安装器均已接入。
- **已有文件复用：** APDrawingGAN 优先复用项目根目录中通过校验的 `APDrawingGAN-Models1.zip`，仅提取所需 checkpoint。GANs N’ Roses 下载固定上游版本的推理源码。
- **下载后的运行条件：** 下载脚本仅准备模型文件。实际推理需要 WSL2 Ubuntu、可用的 PyTorch/CUDA 环境和支持 CUDA 的 NVIDIA 驱动。模型文件不随 Git 分发，使用时请遵守上游许可证与发布条件。

```powershell
# 首次准备环境、下载模型并运行内置示例
# 需提前安装 WSL2 Ubuntu、Linux Python 3.10–3.12（含 venv）和 NVIDIA 驱动
.\scripts\run_models.ps1 -Setup

# 已有环境：使用自己的照片运行两个模型
.\scripts\run_models.ps1 -InputImage "D:\photos\portrait.jpg"

# 已准备模型与环境：启动 Web 后端
.\scripts\start_web.ps1
```

`-Setup` 会下载较大的 PyTorch 包，不能与 `-APDPython` 同时使用；已有自定义 WSL 环境时，先单独下载模型，再用 `-APDPython` 指定解释器运行。普通运行不会自动安装依赖或下载模型。原有 `start_web.ps1 -SetupAPD` **只准备 WSL 环境，不下载 APDrawingGAN 权重**。

单图运行默认将两个模型的 PNG、SVG 和运行记录分别保存到 `outputs/models/apdrawinggan/` 和 `outputs/models/cartoon/`。更多运行参数见 [自动下载与运行说明](docs/AUTO_MODELS.md)。

### GANs N’ Roses 下载过慢：换源或导入已有权重

当前默认使用作者发布的 Google Drive 权重，尚未验证可替代它的公开镜像。若有可访问的 HTTPS 权重直链，可以指定下载源；文件必须与官方推理权重完全一致，脚本仍检查大小和 SHA256。以下地址是占位示例，需替换成实际直链：

```powershell
python scripts/setup_cartoon.py --checkpoint-url "https://你的下载站/GNR_checkpoint.pt"
```

也可在当前 PowerShell 设置 `$env:ROBOT_PORTRAIT_GNR_URL = "https://你的下载站/GNR_checkpoint.pt"`，然后运行 `python scripts/download_models.py --model cartoon`；安装器和 Web 的 `-SetupCartoon` 同样会读取此设置。换源不保证提速，取决于实际网络连接。已有校验通过的本地权重会直接复用。

这台机器已下载过模型时，可把 `third_party/GANsNRoses/GNR_checkpoint.pt` 复制到另一台机器，再导入，无需重新下载约 302 MiB 的权重：

```powershell
python scripts/setup_cartoon.py --checkpoint-file "D:\models\GNR_checkpoint.pt"
```

也支持环境变量 `ROBOT_PORTRAIT_GNR_FILE`。本地文件设置优先于环境变量中的 URL，显式 `--checkpoint-url` 则优先于环境变量；同一次命令不能同时传两个参数。缺少的推理源码仍会从 GitHub 下载，完整离线迁移可复制整个 `third_party/GANsNRoses/` 目录。仅此处的单独下载脚本支持这两个命令行参数，统一安装入口请使用环境变量。

## 开发检查

```powershell
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m pytest -q
Set-Location .\frontend
npm run typecheck
npm run build
```

更多首次安装与故障排查说明见 [`docs/TEAMMATE_QUICKSTART.md`](docs/TEAMMATE_QUICKSTART.md) 和 [`docs/INSTALL_TROUBLESHOOTING.md`](docs/INSTALL_TROUBLESHOOTING.md)。
