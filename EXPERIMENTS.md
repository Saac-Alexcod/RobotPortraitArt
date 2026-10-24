# 实验记录

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

## 11. 后续正式多样本实验

固定单图 benchmark 用于算法调试和消融。

最终报告建议再补：

```text
10～30 张真实人脸样本
```

记录：

- APDrawingGAN 成功率；
- YuNet 检测成功率；
- Stroke count；
- Pen-up；
- Pen-down；
- Vectorize time；
- SVG bytes；
- SSIM；
- Edge-F1；
- 可选 LPIPS；
- eyes / eyebrows / nose / mouth / contour / hair 人工评分。

建议汇总：

```text
mean
std
median
min
max
```

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
