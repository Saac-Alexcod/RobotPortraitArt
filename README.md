# 机器人肖像绘制艺术风格学习（RobotPortraitArt）

本项目面向课程课题“机器人肖像绘制艺术风格学习”，构建从**人脸照片 → 艺术肖像生成 → SVG 矢量化 → 机器人绘制轨迹规划**的完整流程。

项目保留可直接运行的 OpenCV baseline，同时接入 APDrawingGAN，并在 GAN 输出之后实现中心线骨架提取、骨架图构建、最少 trail 覆盖和 Pen-up 空程优化。

> 当前重点不只是“生成一张线稿”，而是进一步得到更适合机器人笔尖执行的有序绘制轨迹。

---

## 1. 当前项目状态

当前主流程已经完成：

- OpenCV 自适应阈值线稿 baseline；
- 统一 `PortraitGenerator` / `GenerationPipeline` 接口；
- APDrawingGAN WSL + CUDA 推理适配；
- APDrawingGAN combined checkpoint 支持；
- YuNet 5 点人脸关键点自动检测；
- 512×512 similarity alignment；
- PPHumanSeg 自动人物 mask；
- APDrawingGAN 自动预处理模式；
- 手动 landmark + mask 模式保留；
- `contour` 轮廓矢量化；
- `centerline` 中心线矢量化；
- Zhang-Suen skeletonization；
- junction cluster contraction；
- Minimum-Lift Trail Cover；
- greedy + deterministic 2-opt Pen-up 路径优化；
- SVG 回栅格；
- SSIM / Edge-F1 / 可选 LPIPS；
- 批量评价与固定 raster benchmark。

当前自动测试：

```text
149 passed, 1 warning
```

唯一 warning 来自主 Windows 环境没有安装可选 PyTorch/LPIPS；APDrawingGAN 的 PyTorch/CUDA 环境独立放在 WSL 中，因此该 warning 不影响主 Pipeline。

---

## 2. 系统流程

```text
输入人脸照片
    ↓
统一预处理
    ↓
Generator
    ├─ baseline
    └─ apdrawinggan
         ↓
      YuNet 5 点检测
         ↓
      Similarity Alignment
         ↓
      PPHumanSeg Mask
         ↓
      WSL + CUDA APDrawingGAN
    ↓
栅格肖像
    ↓
Vectorization
    ├─ contour
    └─ centerline
         ↓
      Zhang-Suen Skeleton
         ↓
      Junction Cluster Contraction
         ↓
      Skeleton Graph
         ↓
      Minimum-Lift Trail Cover
         ↓
      Greedy Ordering
         ↓
      2-opt Pen-up Optimization
    ↓
SVG + Vector Render + Manifest
```

---

## 3. 目录结构

```text
RobotPortraitArt/
├─ configs/
│  ├─ default.yaml              # 默认 contour
│  └─ centerline.yaml           # robot-friendly centerline
├─ data/
│  ├─ input/                    # 私人输入照片，不提交
│  ├─ reference_style/          # 独立参考线稿
│  ├─ CelebA/                   # 本地数据集，不提交
│  └─ apdrawing_preprocessed/   # 可选手动 APDrawingGAN 输入
├─ outputs/                     # 运行结果，不提交
├─ scripts/
│  ├─ run_pipeline.py
│  ├─ check_apdrawinggan.py
│  ├─ benchmark_vectorization.py
│  ├─ prepare_image.py
│  ├─ run_baseline.py
│  ├─ rasterize_svg.py
│  ├─ evaluate.py
│  ├─ batch_evaluate.py
│  └─ ...
├─ src/robot_portrait/
│  ├─ pipeline.py
│  ├─ preprocessing.py
│  ├─ apdrawinggan_preprocess.py
│  ├─ centerline.py
│  ├─ vectorize.py
│  ├─ metrics.py
│  └─ generators/
├─ tests/
├─ third_party/
│  ├─ APDrawingGAN/
│  └─ models/                   # YuNet / PPHumanSeg ONNX（本地）
├─ environment_apdrawinggan.yml
├─ EXPERIMENTS.md
├─ REPORT_DATA_TEMPLATE.csv
├─ requirements.txt
└─ requirements-optional.txt
```

大文件、私人数据和运行产物均由 `.gitignore` 排除，包括：

```text
.venv/
outputs/
data/input/*
data/CelebA/*
APDrawingGAN checkpoints/
*.pt
*.onnx
```

因此本地 checkpoint / ONNX 模型可以存在，但不会被误提交到 Git。

---

## 4. 主工程安装

推荐 Windows 主环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

检查：

```powershell
python -m pytest -q
```

可选评价 / SVG 依赖：

```powershell
python -m pip install -r requirements-optional.txt
```

`requirements-optional.txt` 中的 PyTorch/LPIPS 仅用于可选 LPIPS 指标。APDrawingGAN 推理使用独立 WSL Python，不要求把 PyTorch 装入 Windows 主环境。

---

## 5. 统一 Pipeline

### 5.1 查看当前生成器状态

```powershell
python scripts/run_pipeline.py --list-methods
```

APDrawingGAN 使用指定 WSL Python：

```powershell
python scripts/run_pipeline.py `
  --list-methods `
  --apd-python /home/winter/.venvs/robotportrait-apd/bin/python
```

也可以在当前 PowerShell 设置：

```powershell
$env:ROBOT_PORTRAIT_APD_PYTHON = "/home/winter/.venvs/robotportrait-apd/bin/python"
```

之后直接：

```powershell
python scripts/run_pipeline.py --list-methods
```

---

### 5.2 Baseline + Contour

```powershell
python scripts/run_pipeline.py `
  --input "data/input/face.png" `
  --output-dir "outputs/demo_baseline_contour" `
  --config configs/default.yaml `
  --method baseline `
  --size 512
```

---

### 5.3 Baseline + Centerline

```powershell
python scripts/run_pipeline.py `
  --input "data/input/face.png" `
  --output-dir "outputs/demo_baseline_centerline" `
  --config configs/centerline.yaml `
  --method baseline `
  --size 512
```

---

### 5.4 APDrawingGAN 自动模式

自动模式不需要手动传 `--landmark` / `--mask`：

```powershell
python scripts/run_pipeline.py `
  --input "data/input/face.png" `
  --output-dir "outputs/demo_apdrawinggan" `
  --config configs/centerline.yaml `
  --method apdrawinggan `
  --size 512 `
  --apd-python /home/winter/.venvs/robotportrait-apd/bin/python `
  --apd-timeout 180
```

自动预处理流程：

```text
OpenCV YuNet
→ 5-point landmarks
→ APDrawingGAN similarity alignment (512)
→ OpenCV Zoo PPHumanSeg
→ APDrawingGAN
```

运行结束后会生成：

```text
prepared.png
apdrawinggan.png
apdrawinggan.svg
apdrawinggan_vector_render.png
manifest.json
apdrawinggan_preprocess/
```

---

### 5.5 APDrawingGAN 手动模式

如果要复现实验或使用已有 landmark / mask，可同时提供：

```powershell
python scripts/run_pipeline.py `
  --input "data/input/face.png" `
  --output-dir "outputs/demo_apdrawinggan_manual" `
  --config configs/centerline.yaml `
  --method apdrawinggan `
  --size 512 `
  --landmark "data/apdrawing_preprocessed/face.txt" `
  --mask "data/apdrawing_preprocessed/face_mask.png" `
  --apd-python /home/winter/.venvs/robotportrait-apd/bin/python
```

`--landmark` 和 `--mask` 必须同时提供；两者都省略时进入自动模式。

---

## 6. Windows + WSL APDrawingGAN 环境

APDrawingGAN legacy 环境建议保持独立，例如：

```text
WSL distro:
Ubuntu-22.04

Python:
/home/winter/.venvs/robotportrait-apd/bin/python
```

检查 APDrawingGAN 环境：

```powershell
python scripts/check_apdrawinggan.py `
  --python /home/winter/.venvs/robotportrait-apd/bin/python
```

或：

```powershell
python scripts/run_pipeline.py `
  --list-methods `
  --apd-python /home/winter/.venvs/robotportrait-apd/bin/python
```

### H: 盘自动挂载

如果项目位于 `H:`，而 WSL 冷启动后无法访问 `/mnt/h`，可以在 WSL `/etc/fstab` 中配置：

```text
H: /mnt/h drvfs defaults,nofail 0 0
```

之后：

```powershell
wsl --shutdown
wsl -d Ubuntu-22.04 --cd ~ -- findmnt /mnt/h
```

即可验证冷启动自动挂载。

---

## 7. APDrawingGAN 本地资源

项目代码支持 APDrawingGAN combined checkpoint：

```text
third_party/APDrawingGAN/checkpoints/formal_author/300_net_gen.pt
```

以及自动预处理模型：

```text
third_party/models/face_detection_yunet_2023mar.onnx
third_party/models/human_segmentation_pphumanseg_2023mar.onnx
```

这些模型文件被 `.gitignore` 排除，不应提交进 Git。新环境需要按照项目说明自行准备。

APDrawingGAN 上游：

```text
https://github.com/yiranran/APDrawingGAN
```

主工程不修改第三方模型核心代码。

---

## 8. Vectorization

### 8.1 Contour 模式

默认配置：

```text
configs/default.yaml
```

主要特点：

- OpenCV contour extraction；
- 路径数量少；
- 速度快；
- 对粗线更接近区域外边界；
- 适合作为兼容 baseline。

---

### 8.2 Centerline 模式

配置：

```text
configs/centerline.yaml
```

流程：

```text
Raster
→ Threshold
→ Zhang-Suen Skeletonization
→ Spur Pruning
→ Junction Cluster Contraction
→ Graph Stroke Extraction
→ Minimum-Lift Trail Cover
→ Greedy Ordering
→ 2-opt
→ SVG
```

核心目标是获得更接近真实笔尖运动的开放中心线轨迹。

---

## 9. 路径规划实验

为了避免 APDrawingGAN 多次 GPU 推理造成输入差异，路径实验固定使用同一张 GAN raster：

```text
SHA256:
13A35D6DC7E0E8831E8C0DBAD6ED0714C0ACA0BEF0F75FCCD7A5439A6ED696ED
```

消融结果：

| 阶段 | Stroke | Pen-up / px | Pen-down / px | 说明 |
|---|---:|---:|---:|---|
| Graph Refinement | 713 | 5980.14 | 15592.30 | junction contraction + local merge |
| Minimum-Lift | 431 | 5665.22 | 15602.68 | theoretical minimum trail cover |
| 2-opt | **431** | **5001.77** | **15602.68** | inter-trail travel optimization |

最终：

```text
Stroke:
713 → 431
减少约 39.55%

Pen-up:
5980.14 → 5001.77 px
减少约 16.36%

Minimum-Lift:
431 = theoretical minimum trail count
edge_reuse_count = 0
```

E2.2 → E2.3 中 Pen-down 长度保持完全不变，说明 2-opt 的收益来自轨迹访问顺序，而不是删除绘制内容。

E2.2 / E2.3 回栅格比较：

```text
different_pixels = 1770 / 262144 ≈ 0.675%
max_channel_diff = 3 / 255
mean_abs_diff    = 0.007061
```

仅存在极轻微的回栅格抗锯齿差异，视觉几何基本一致。

---

## 10. 固定 Raster Benchmark

使用：

```powershell
python scripts/benchmark_vectorization.py `
  --input "outputs/vector_fixed_benchmark/input.png" `
  --output-dir "outputs/vector_fixed_benchmark/current"
```

会同时输出：

```text
contour.svg
contour_render.png
centerline.svg
centerline_render.png
comparison.json
```

`comparison.json` 中记录：

- `stroke_count`
- `raw_stroke_count`
- `pen_down_length_px`
- `pen_up_length_px`
- `junction_cluster_count`
- `theoretical_min_trail_count`
- `edge_reuse_count`
- `two_opt_improvement_px`
- `vectorize_ms`
- `svg bytes`

---

## 11. 评价

单图：

```powershell
python scripts/evaluate.py `
  --pred outputs/pred.png `
  --ref data/reference_style/face.png `
  --json_out outputs/metrics.json
```

支持：

- SSIM；
- Edge-F1；
- 可选 LPIPS。

没有独立参考线稿时，不应将 PNG → SVG → PNG 的保真度评价称为“艺术风格质量”。

批量评价：

```powershell
python scripts/batch_evaluate.py `
  --reference-dir data/reference_style `
  --method baseline=outputs/baseline `
  --method apdrawinggan=outputs/apdrawinggan `
  --output-csv outputs/metrics.csv `
  --summary-json outputs/summary.json `
  --svg-dir outputs/svg
```

人工局部评分建议继续保留：

- eyes；
- eyebrows；
- nose；
- mouth；
- contour；
- hair。

---

## 12. 测试

完整测试：

```powershell
python -m pytest -q
```

当前：

```text
149 passed, 1 warning
```

中心线 / 路径规划专项测试覆盖：

- thick / diagonal / L / T / loop skeleton；
- spur pruning；
- junction clustering；
- stroke tracing；
- minimum-lift trail cover；
- Euler / odd-degree cases；
- edge reuse；
- deterministic 2-opt；
- Pen-up 不退化；
- SVG open-path rasterization。

---

## 13. 当前限制

### 13.1 APDrawingGAN GPU 推理并非严格 deterministic

已观察到：

```text
prepared / landmark / alignment / mask 相同
```

但两次 APDrawingGAN GPU 输出的 PNG SHA256 可能不同。

因此严格的 vectorization 消融使用固定 raster 输入，而不是重复运行 GAN。

后续若需要进一步研究，可检查：

- Torch / NumPy / Python seed；
- `torch.use_deterministic_algorithms`；
- cuDNN deterministic；
- CUDA 算子确定性；
- APDrawingGAN 内部随机操作。

### 13.2 Centerline 已达到当前图结构下的 trail 下界

当前固定 raster：

```text
odd_vertex_count = 862
minimum trails   = 431
actual trails    = 431
```

继续减少 trail 数将需要改变图本身，例如：

- 删除分支；
- 修复断线；
- 允许短距离跨空白；
- 允许少量重复描边。

这些属于视觉质量与机器人效率之间的新权衡，不建议在没有明确 failure case 时继续盲目优化。

---

## 14. 下一阶段

当前核心算法已经稳定，下一阶段优先提升系统完整度：

1. README / EXPERIMENTS / 报告数据同步；
2. FastAPI Backend；
3. Vue 3 + TypeScript Web Demo；
4. Robot Drawing Animation；
5. 10～30 张多样本 benchmark；
6. 最终报告与 PPT；
7. 有硬件条件时再增加机器人/G-code exporter。

Web 端建议支持：

```text
上传人脸
→ 选择 baseline / APDrawingGAN
→ 选择 contour / centerline
→ Generate
→ Original / Prepared / GAN / Vector 对比
→ Stroke / Pen-up / Timing 指标
→ 绘制动画
→ 下载 SVG / PNG / Manifest
```

---

## 15. 实验与课程材料

真实实验记录见：

```text
EXPERIMENTS.md
```

报告数据表：

```text
REPORT_DATA_TEMPLATE.csv
```

个人贡献与后续规划可继续从实验结果中整理，不应填写未经实际运行得到的数据。
## Documentation / 文档

- [Final Project Report / 最终项目报告](docs/FINAL_REPORT.md)
