# Optional APDrawingGAN setup

For a fresh Windows computer, begin with [First-run setup and networking](FIRST_RUN_SETUP.md). It provides read-only diagnostics and opt-in, integrity-checked downloads.

本说明面向希望从 **OpenCV Baseline** 升级到 **APDrawingGAN 自动预处理 + WSL/CUDA 推理** 的使用者。无需 GPU 的最小上手流程请先阅读仓库根目录 `README.md`。

## 1. Why weights are not included

仓库中保存算法适配代码、配置和第三方源码，但不分发以下大文件：

- APDrawingGAN 预训练生成器 checkpoint；
- YuNet / PPHumanSeg ONNX 权重；
- CelebA 数据集、人像私有数据及历史运行输出。

这有利于减小仓库体积，并尊重第三方权重或数据的再分发条件。**下载源代码不会自动配置 GPU 推理功能。**

## 2. Obtain official model files

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
# 按你的实际 WSL 发行版与 Python 位置替换参数
.\scripts\start_web.ps1 `
  -Distro Ubuntu-22.04 `
  -APDPython /home/you/.venvs/robotportrait-apd/bin/python
```

启动后打开 `http://127.0.0.1:8000/api/methods`，同时确认：

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
