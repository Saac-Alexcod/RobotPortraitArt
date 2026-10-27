# Optional APDrawingGAN setup

新组员先看 [一页上手指南](TEAMMATE_QUICKSTART.md)，使用功能分支的 `install.ps1` 尝试自动安装和模型校验；本文件提供手动准备模型与环境的补充说明。

本说明面向希望从 **OpenCV Baseline** 升级到 **APDrawingGAN 自动预处理 + WSL/CUDA 推理** 的使用者。无需 GPU 的最小上手流程请先阅读仓库根目录 `README.md`。

## 1. Why weights are not included

仓库中保存算法适配代码、配置和第三方源码，但不分发以下大文件：

- APDrawingGAN 预训练生成器 checkpoint；
- YuNet / PPHumanSeg ONNX 权重；
- CelebA 数据集、人像私有数据及历史运行输出。

这有利于减小仓库体积，并尊重第三方权重或数据的再分发条件。**仅下载源码不会自动获得模型或 CUDA；需执行功能分支中的 `install.ps1` 尝试自动准备，或按本文件手动配置。**

## 2. Obtain official model files

可运行 `python scripts/download_models.py --model apdrawinggan` 自动下载并安装以下三个权重；下载两个模型时省略 `--model`。运行入口与参数见 [自动下载与运行说明](AUTO_MODELS.md)。

| Resource | Upstream source | Local path |
|---|---|---|
| APDrawingGAN pre-trained models | [APDrawingGAN README](https://github.com/yiranran/APDrawingGAN)（上游文档的 Model1） | `third_party/APDrawingGAN/checkpoints/formal_author/` |
| YuNet face detector | [OpenCV Zoo / YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | `third_party/models/face_detection_yunet_2023mar.onnx` |
| PPHumanSeg human segmentation | [OpenCV Zoo / PPHumanSeg](https://github.com/opencv/opencv_zoo/tree/main/models/human_segmentation_pphumanseg) | `third_party/models/human_segmentation_pphumanseg_2023mar.onnx` |

此项目已针对官方 APDrawingGAN `formal_author` 的 epoch 300 输出做适配。用户此前所用的是适配器支持的合并 checkpoint 文件 `300_net_gen.pt`；具体权重布局需以适配器的预检反馈为准，不要把下载的 ZIP 或 Git LFS 指针文件误当作实际模型。

请从上游项目页面下载并确认其版权及许可条款。不应通过 GitHub 提交 checkpoint、ONNX 或从 CelebA 解压出来的人脸图片。

## 3. Environment

- Windows + WSL2（示例使用 `Ubuntu-22.04`）；
- 适配 APDrawingGAN 旧代码的独立 Python 环境；
- PyTorch / torchvision 与当前 NVIDIA 驱动和 CUDA 兼容；
- 上游 APDrawingGAN 所需的其他依赖。

由于 APDrawingGAN 是较早版本的研究代码，其依赖版本可能对当前 PyTorch 需要额外兼容处理。请参考 `third_party/APDrawingGAN/readme.md` 及项目适配器报错逐项确认，不应直接断言最新 Python/PyTorch 组合都适用。

示例（在项目根目录的 PowerShell）：

```powershell
# 自动选择发行版及已有 CUDA Python 环境
.\scripts\start_web.ps1

# 首次安装独立 WSL 环境，之后普通启动会自动找到它
.\scripts\start_web.ps1 -SetupAPD

# 按你的实际 WSL 发行版与 Python 位置替换参数
.\scripts\start_web.ps1 `
  -Distro Ubuntu-22.04 `
  -APDPython /home/you/.venvs/robotportrait-apd/bin/python
```

未传 `-APDPython` 时，脚本优先读取 `ROBOT_PORTRAIT_APD_PYTHON` 环境变量；未配置则扫描 WSL 当前用户的常见 venv/Conda 环境和系统 Python，选择依赖及 CUDA 运算检查通过的解释器。`-Distro` / `ROBOT_PORTRAIT_APD_DISTRO` 可覆盖自动选择的发行版。自定义路径使用 Linux 绝对路径，无需修改脚本。

`-SetupAPD` 只管理 WSL 当前用户的 `~/.venvs/robotportrait-apd`，不会向系统 Python 安装包。此安装配置支持 WSL Python 3.10–3.12，使用 PyTorch 2.5.1 / torchvision 0.20.1 的 CUDA 12.1 包，并安装推理所需的 NumPy、Pillow、dominate 和 OpenCV。它是一个固定的兼容配置，不能保证支持所有显卡；若 GPU/驱动不兼容，请自行准备合适的环境并用 `-APDPython` 指定。安装会先检查 `nvidia-smi`，最后执行 CUDA 运算检查；失败会停止，修复原因后可重试。

需要提前安装 WSL 发行版及支持 WSL CUDA 的 NVIDIA Windows 驱动。若提示缺少 venv，在所选 Ubuntu 的终端执行 `sudo apt update`、`sudo apt install python3-venv`，然后重试 `-SetupAPD`。该选项不能与 `-CheckOnly` 或显式 `-APDPython` 同用；初始化时忽略继承的 `ROBOT_PORTRAIT_APD_PYTHON`，始终安装到托管环境。普通启动仍优先采用环境变量，如需恢复自动检测，可在 PowerShell 执行 `Remove-Item Env:ROBOT_PORTRAIT_APD_PYTHON -ErrorAction SilentlyContinue`。

安装包需要 WSL 能访问 PyPI 和 PyTorch 下载站。出现 localhost 代理警告且下载失败时，请先让 WSL 使用可访问的代理地址或直连，再重试；脚本不修改全局 WSL 网络设置。仅出现代理警告不代表本地 Python 无法运行。

如果已有 `bin/python` 但提示 `No module named ensurepip` / `pip`，可能是上次创建环境中断。脚本会检测 pip 并尝试补齐环境，不删除已有包。若系统缺少组件，会提示与当前 Python 次版本匹配的 Ubuntu 包；例如 Python 3.10 可在 PowerShell 执行：

```powershell
wsl -d Ubuntu-22.04 -- sudo apt-get update
wsl -d Ubuntu-22.04 -- sudo apt-get install -y python3.10-venv
.\scripts\start_web.ps1 -SetupAPD
```

`sudo` 可能要求输入 WSL 用户密码。其他发行版应替换发行版名称，并按脚本提示安装对应的 venv 包。

`-SetupAPD` 不下载 checkpoint、YuNet 或 PPHumanSeg 模型，仍需按上表准备。`-CheckOnly` 仅检查环境及可用端口，不安装包、不启动服务；要同时验证完整 GAN 可用性，可运行 `.\scripts\start_web.ps1 -CheckOnly -RequireAPDrawingGAN`。

启动后打开终端打印的 API 地址加 `/api/methods`（8000 被占用时会自动换端口），同时确认：

1. `apdrawinggan.available` 是 `true`；
2. `preflight.ready` 是 `true`；
3. `auto_preprocess.available` 是 `true`；
4. YuNet/PPHumanSeg 文件均实际存在，而不只是同名的 Git LFS 占位符。

如果环境暂不可用，继续使用 OpenCV Baseline。只有明确要求 GAN 可用才使用 `-RequireAPDrawingGAN` 严格启动模式。

## 4. Optional CelebA benchmark

CelebA 数据集并不是本项目“上传一张图生成肖像”的必需条件。Benchmark 场景如果要使用它，需要自行从数据集官方页面下载，并遵守非商业研究与教育限制以及第三方图片版权条件。

- 数据集说明：`data/CelebA/README.txt`
- 数据集主页：[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

不要在仓库中重新托管原始分卷或人物照片。
