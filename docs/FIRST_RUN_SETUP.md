# 首次运行、模型准备与人工排障入口

> 适用于首次接触 RobotPortraitArt 的组员。**先走自动安装入口**，只有自动下载或环境准备失败时才按照下面的手动方式处理。

## 1. 首次安装（推荐）

自动安装器已经合并到默认 `main` 分支。在 Windows PowerShell 中执行：

```powershell
git clone --branch anime-addition https://github.com/Saac-Alexcod/RobotPortraitArt.git
cd RobotPortraitArt
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
```

**组员只需复制命令、查看验收与故障处理**：请使用 [一页上手指南](TEAMMATE_QUICKSTART.md)。安装器当前执行 7 阶段检查/准备：Windows Python → Windows 依赖 → Node/npm → WSL2 → Linux CUDA PyTorch → 模型 → 真实 GAN 与 SVG 推理；不使用 `-NoStart` 则继续尝试启动两个网页服务。

本仓库**未提交** `.venv`、`node_modules`、ONNX、APDrawingGAN checkpoint 及 CelebA 数据集。首次完整 GAN 安装需要足够磁盘空间、访问各官方资源的网络、WSL2 和支持的 NVIDIA GPU/驱动；某些系统条件需要管理员授权或重启。**如果 `wsl.exe` 完全不存在，当前安装器会明确失败，但不会自动替用户启用 Windows WSL 可选功能。**

## 2. 如何看到真正成功的结果

```text
[PASS] First-run acceptance completed.
```

除此之外，必须检查 `outputs/first_run_validation/` 中的真实 PNG、SVG、`manifest.json`，浏览器访问 `http://127.0.0.1:5173`，选择 APDrawingGAN + Centerline 完成一次任务。只有网页启动但没有完成 GAN 任务，不算最终用户验收。

若采用 `-NoStart`，会执行真实 GAN 检测但**不启动网页**；`-BaselineOnly` 可在无 CUDA 机器运行 OpenCV 对照方案，但**不代表 GAN 部署完成**。

## 3. 重启、重复执行和网络中断

Windows 提示 UAC / 重启 / Ubuntu 首次用户初始化时，先完成操作，然后返回克隆目录重复执行**相同安装命令**。已有有效组件会尽可能复用。中断后的 `.part` 只是暂存下载，不要自行更名为模型文件；官方网络端点可能不支持精确断点续传。

可选的代理命令（请替换真实端口）：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -Proxy 'http://127.0.0.1:7890'
```

WSL NAT 下 localhost 代理警告通常不妨碍本地推理；但如果 WSL 内 `pip` 或模型源真的不可达，需要额外排查 WSL 网络和代理监听。请勿在命令历史、Git 或聊天中公开代理口令。完整指引：[故障排查手册](INSTALL_TROUBLESHOOTING.md)。

## 4. 自动下载失败时，模型文件的官方来源

| 资源 | 官方来源 | 项目内期望位置 |
|---|---|---|
| YuNet | [OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | `third_party/models/face_detection_yunet_2023mar.onnx` |
| PPHumanSeg | [OpenCV Zoo PPHumanSeg](https://github.com/opencv/opencv_zoo/tree/main/models/human_segmentation_pphumanseg) | `third_party/models/human_segmentation_pphumanseg_2023mar.onnx` |
| APDrawingGAN Model1 | [APDrawingGAN](https://github.com/yiranran/APDrawingGAN) | `third_party/APDrawingGAN/checkpoints/formal_author/300_net_gen.pt`（合并权重布局） |

只从官方/可信来源下载并遵守权重许可，不要把下载的 ZIP、HTML 错误页面、Git LFS 指针文件伪装成 `.pt`/`.onnx`。具体 checkpoint 布局及预检方法见 [模型说明](MODEL_SETUP.md)。**CelebA 不是运行内置示例必需品**，无需为首次部署下载庞大的数据集。

## 5. 有经验的成员可用的高级操作（可选）

```powershell
# 检查已安装 WSL 候选，要求存在可用 Windows Python
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -CheckWSLOnly

# 只配置/验收，不启动前后端
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -NoStart

# 显式复用已经验证的 WSL Python；路径必须是你电脑上的，不是作者用户名
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -Distro Ubuntu-22.04 -APDPython '/home/YOUR_USER/.venvs/robotportrait-apd/bin/python'
```

保留的手动辅助脚本 `scripts/setup_environment.ps1`、`scripts/doctor_environment.py` 是**故障诊断/分步下载工具**，不是组员首先应尝试的正常安装路线。Windows 的后端 Python 与 Ubuntu 的 GAN Python 是两套独立环境；不要因为 Windows 虚拟环境没有 `torch` 就随意安装或重装 WSL CUDA。

## 文档导航

- [组员一页上手](TEAMMATE_QUICKSTART.md)：直接复制命令。
- [自动安装器执行机制与限制](AUTOMATIC_INSTALL.md)：7 个阶段与真实验收。
- [故障日志与错误码](INSTALL_TROUBLESHOOTING.md)：网络、WSL、CUDA、版本冲突。
- [模型与许可证](MODEL_SETUP.md)：上游下载来源和文件要求。
