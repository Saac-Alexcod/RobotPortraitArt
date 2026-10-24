# 固定栅格与路径优化阶段性实验记录（E0–E5）

> **文档定位：** 这是路径规划开发阶段的实验日志，不是最新软件测试清单，也不是完整的 20 张人脸 Benchmark 报告。文中的 `149 passed, 1 warning` 为当时的测试快照，后续版本的回归数据不得覆盖此历史记录。
> **对应正式研究：** 20 张 Core / 40 次端到端评估与 18 对结构保真度结果见 [技术实现与实验分析报告](docs/FINAL_REPORT.md)。
> **公开样例区别：** `data/examples/` 的 4 张合成演示照片不属于上述正式 Benchmark。
> **证据边界：** 本页的固定 raster SHA256、stroke、pen-up 数值取自开发阶段记录，未在本次文档校订中重新计算。

本文件只记录**真实运行结果**。未经实际程序运行的指标不得填写。

---

## 1. 环境与实现状态

### 主工程

- Windows 主 Python：现代 Python `.venv`
- OpenCV baseline：已完成
- 统一 `GenerationPipeline`：已完成
- Contour SVG：已完成
- Centerline SVG：已完成
- 自动测试：`149 passed, 1 warning`

### APDrawingGAN

- 上游：`yiranran/APDrawingGAN`
- WSL：`Ubuntu-22.04`
- legacy Python：`/home/winter/.venvs/robotportrait-apd/bin/python`
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU
- Torch：`2.5.1+cu121`
- CUDA：可用
- Combined checkpoint：`300_net_gen.pt`
- 自动 landmark：YuNet
- 自动 alignment：512×512 similarity transform
- 自动 mask：PPHumanSeg
- 自动 Pipeline：已完成
- 手动 landmark + mask：保留

---

## 2. E0 — OpenCV Baseline

### 目的

验证：

```text
preprocess
→ baseline line art
→ vectorization
→ SVG
→ rasterize
→ manifest
```

完整流程可运行。

### 状态

已完成 smoke test。

### 说明

E0 用作：

- 可运行 fallback；
- Generator abstraction 对照；
- 无 GPU 环境演示；
- 与 APDrawingGAN 的视觉基线比较。

正式艺术质量评价仍需要独立参考线稿。

---

## 3. E1 — APDrawingGAN 自动肖像生成

### 目的

验证普通照片可以在不手动提供 landmark / mask 的情况下直接进入 APDrawingGAN。

### 自动预处理

```text
Input Photo
→ OpenCV YuNet
→ 5-point Landmark
→ Similarity Alignment 512
→ PPHumanSeg Mask
→ APDrawingGAN
```

### 已验证内容

- APDrawingGAN source preflight；
- combined checkpoint；
- WSL Python；
- PyTorch modules；
- CUDA；
- GPU；
- YuNet；
- alignment；
- PPHumanSeg；
- APDrawingGAN 推理；
- SVG / render / manifest。

### 示例真实运行

此前真实单图 Pipeline 已完成，APDrawingGAN 推理时间为秒级。

注意：多次 GPU 推理即使预处理输入一致，最终 PNG SHA256 仍可能不同，因此严格 vector benchmark 不重复执行 GAN。

---

## 4. E2 — Contour Vectorization

### 配置

```text
configs/default.yaml
```

### 固定 Raster

为保证 vectorization A/B 公平，后续实验统一使用同一张 APDrawingGAN raster：

```text
SHA256:
13A35D6DC7E0E8831E8C0DBAD6ED0714C0ACA0BEF0F75FCCD7A5439A6ED696ED
```

### 实验结果

```text
mode                = contour
stroke_count        = 88
closed_stroke_count = 88
pen_down_length_px  = 9768.5156
pen_up_length_px    = 2629.5712
svg_bytes           = 25099
```

说明：

- 路径数量少；
- 矢量化速度快；
- 粗线主要表现为区域边界；
- 可作为默认兼容模式。

---

## 5. E3 — Centerline Graph Refinement

### 目标

解决真实 skeleton 中：

- 多像素 junction；
- 头发区域碎片化；
- 极短 graph edge；
- 过多 stroke。

### 方法

```text
Threshold
→ Zhang-Suen Skeletonization
→ Spur Pruning
→ Junction Cluster Contraction
→ Graph Stroke Trace
→ Local Continuation Merge
```

### 固定 Raster 结果

```text
stroke_count        = 713
raw_stroke_count    = 1286
merge_count         = 526
skeleton_pixels     = 15828
junction_pixels     = 2176
junction_clusters   = 492
pen_down_length_px  = 15592.2992
pen_up_length_px    = 5980.1392
```

### 结论

Junction contraction 可以显著降低由多像素交叉点造成的图结构碎片化，但单纯局部 merge 仍然不是全局最少 Pen-lift 方案。

---

## 6. E4 — Minimum-Lift Trail Planning

### 目标

在：

- 不漏画真实 graph edge；
- 不跨空白连接；
- 不重复真实 edge；

的前提下，用最少连续 trail 覆盖 skeleton graph。

### 方法

```text
Skeleton Graph
→ Odd Degree Analysis
→ Temporary Super Node
→ Hierholzer Traversal
→ Cut Virtual Edges
→ Edge-disjoint Trails
```

### 固定 Raster 结果

```text
connected_component_count   = 89
odd_vertex_count            = 862
theoretical_min_trail_count = 431
actual_stroke_count         = 431
edge_reuse_count            = 0
pen_down_length_px          = 15602.6828
pen_up_length_px            = 5665.2158
```

### 理论验证

```text
862 odd vertices
→ 862 / 2
→ 431 minimum open trails
```

实际结果：

```text
431 = theoretical minimum
```

说明当前图在“不重复 edge”的约束下已经达到最少 trail 数。

---

## 7. E5 — 2-opt Pen-up Route Optimization

### 目标

E4 已无法继续降低 trail 数，因此只优化 431 条固定 trail 的访问顺序和方向。

### 方法

```text
Minimum-Lift Trails
→ Greedy Initial Order
→ Deterministic Open-path 2-opt
→ Optimized Pen-up Route
```

### 固定 Raster 结果

```text
stroke_count                      = 431
theoretical_min_trail_count       = 431
edge_reuse_count                  = 0

pen_up_greedy_px                  = 5665.2158
pen_up_optimized_px               = 5001.7720
two_opt_improvement_px            = 663.4439
two_opt_improvement_ratio         = 0.1171083

pen_down_length_px                = 15602.6828
two_opt_move_count                = 16
two_opt_pass_count                = 16
```

### 结论

E4 → E5：

```text
Pen-up:
5665.22 → 5001.77 px
下降约 11.71%

Stroke:
431 → 431
不变

Pen-down:
15602.68 → 15602.68
不变

Edge reuse:
0 → 0
```

因此 2-opt 的收益来自机器人空程顺序优化，而不是减少肖像内容。

---

## 8. E3 → E5 总体改进

```text
Stroke:
713 → 431
减少约 39.55%

Pen-up:
5980.14 → 5001.77
减少约 16.36%
```

---

## 9. E4 / E5 视觉一致性

回栅格逐像素比较：

```text
shape_equal      = True
pixel_equal      = False
different_values = 5310
different_pixels = 1770
max_channel_diff = 3
mean_abs_diff    = 0.0070610
```

总像素：

```text
512 × 512 = 262144
```

不同像素比例：

```text
1770 / 262144 ≈ 0.675%
```

平均绝对差相对 255 满量程约为：

```text
0.00277%
```

结论：

> E5 保持相同 Pen-down 总长度与真实绘制结构，仅因路径顺序不同，在 OpenCV fallback 回栅格时出现极轻微抗锯齿差异，整体视觉几何基本一致。

---

## 10. 固定 Raster Benchmark 命令

```powershell
python scripts/benchmark_vectorization.py `
  --input "outputs/vector_fixed_benchmark/input.png" `
  --output-dir "outputs/vector_fixed_benchmark/current"
```

输出：

```text
contour.svg
contour_render.png
centerline.svg
centerline_render.png
comparison.json
```

---

## 11. 已完成的多样本正式实验与仍待拓展的工作

本文件的 E0–E5 主要使用**固定单图栅格**，用于调试、图结构消融与验证 2-opt 的代价改善。此后项目已在另一冻结实验提交 `0f0426a` 上记录 Core 20 张、两种方法共 40 次端到端运行，成功数为 20/20 和 18/20；其中严格配对比较使用共同完成的 18 个输入。

正式报告同时记录了 Regional Edge-F1 的自动结构保真度分析，但尚**没有**逐图独立艺术家风格真值、多人盲评或真实机械臂运行数据。因此保留如下后续研究任务，而非把它们写成已完成：

- 在**独立**艺术参考图条件下评价 SSIM / LPIPS 与主观艺术性；
- 扩大包含姿态、光线及跨设备输入的样本规模，提供方差和置信区间；
- 结合真实执行器的时间、抬笔高度及运动学约束测量路径代价；
- 用小规模多人匿名偏好调查补充自动 Edge-F1 的解释局限。

20 张样本的设置、失败保留和配对统计详见 [正式报告](docs/FINAL_REPORT.md)，请不要把本文件的单张 E3–E5 数据与正式 Benchmark 的多样本均值混用。

---

## 12. 当前限制

### APDrawingGAN Determinism

已确认：

```text
prepared
landmarks
alignment
mask
```

可以保持一致，但重复 GPU 推理的 PNG hash 可能不同。

因此：

- GAN 质量实验：允许真实多次运行并记录 seed / environment；
- Vectorization 消融：必须固定同一 raster。

### 继续减少 431 trails

当前 E4 已达到当前 graph 理论下界。

若进一步减少 Pen-lift，需要改变图或约束，例如：

- 修复断线；
- 合并近邻端点；
- 允许短距离空白连接；
- 允许少量重复描边；
- 删除不重要的小分支。

这些属于新的视觉质量 / 机器人效率权衡，暂不作为当前核心算法继续扩展。
