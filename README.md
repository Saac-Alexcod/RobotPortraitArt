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

## 开发检查

```powershell
\.venv\Scripts\python.exe -m pip install pytest
\.venv\Scripts\python.exe -m pytest -q
Set-Location .\frontend
npm run typecheck
npm run build
```

更多首次安装与故障排查说明见 [`docs/TEAMMATE_QUICKSTART.md`](docs/TEAMMATE_QUICKSTART.md) 和 [`docs/INSTALL_TROUBLESHOOTING.md`](docs/INSTALL_TROUBLESHOOTING.md)。
