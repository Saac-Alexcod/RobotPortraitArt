# RobotPortraitArt · 2026-10-24 课程项目交接记录（历史快照）

> **历史记录：** 本页反映 2026-10-24 某一开发时点的交接与答辩准备状态，不代表后来发布的 `v1.0.1` 的最新功能清单或测试总数。旧的 `177 passed` 与当时的“仍需验证”按历史事实保留。
> **当前参考：** [公开 README](../README.md)、[技术实验报告](FINAL_REPORT.md)、[版本与证据口径](DOCUMENTATION_STATUS.md)。
> 面向报告/PPT/答辩团队。以代码实际实现与已验证输出为依据，不把模拟轨迹声称为实体机械臂实测。

## 1. 一句话简介

输入人脸照片后，系统通过自动人脸预处理和 APDrawingGAN 生成艺术肖像，随后将栅格画面转换为有序 SVG 笔画，通过路径优化减少抬笔空程，最后在浏览器中按真实 SVG 路径逐笔模拟绘制。

## 2. 主要贡献（PPT 可直接使用）

1. 将 YuNet 五点检测、Similarity Alignment 和 PPHumanSeg 自动预处理集成至 APDrawingGAN 的 WSL/CUDA 推理调用。
2. 设计生成器抽象层，同时保留 OpenCV Baseline 对照。
3. 实现 Centerline 骨架图 → Minimum-Lift Trail Cover → Greedy + 2-opt 行程优化；同时保留 Contour 对照。
4. 搭建 FastAPI 持久化任务接口、串行 GPU Worker、阶段结果文件与 Manifest 指标。
5. Vue 3 提供五阶段结果、图像 A/B 对比和 SVG Path 逐笔回放。

## 3. 真正执行的数据链

```text
人脸照片 → Windows FastAPI POST /api/jobs → JobStore → 单 Worker
→ Pipeline.preprocess → YuNet / Similarity Alignment / PPHumanSeg
→ WSL PyTorch CUDA APDrawingGAN（或 OpenCV Baseline）
→ Portrait PNG → Contour / Centerline → 轨迹规划 → SVG
→ Manifest + Vector Render → GET job/manifest/files → Vue 结果/动画
```

> 注意：Web 模式下 APDrawingGAN 自动依赖 YuNet/PPHumanSeg；`/api/methods` 的可用性不仅要求 CUDA/checkpoint，也要求自动预处理模型就绪。

## 4. 已验证与待验证

- 已有实机反馈（10 月 24 日）：Windows + Ubuntu-22.04 + RTX 4060，APDrawingGAN/Centerline/512 的真实任务 `completed`，结果展示与 Manifest 可读；截图中该样例有 404 笔、笔画绘制距离约 15,627.6 px、空程约 5,563.0 px、总任务约 8.38 s。这是**单张样例**，不要写成全数据集平均。
- 当前环境 Python 回归：177 passed, 1 LPIPS optional warning。
- 当前最终版新增了任务恢复、前端断线处理与 UI 展示，仍需在本人 Windows 环境执行前端 typecheck/build 和实际浏览器回归。
- 无实体机械臂硬件闭环数据；Web 绘制动画是 **SVG 路径软件模拟**。
- 样例人物的头发/眼镜位置出现碎线是当前质量局限，不应声称所有人像均高保真。

## 5. 指标定义（避免答辩误述）

| 指标 | 含义 | 注意 |
| --- | --- | --- |
| Stroke Count | SVG 的笔画条数 | 不能直接说明“越少越好” |
| Pen-down / px | 落笔笔画长度总和 | 基于 2D 图片像素单位，并非机械臂物理长度 |
| Pen-up / px | 相邻轨迹之间抬笔移动距离 | 越低通常表示轨迹访问顺序更省空程 |
| 2-opt 优化率 | (Greedy 初始空程 - 优化后空程) / Greedy 初始空程 | 只反映空程排序改进，不代表画质提升 |
| Vectorize / Total Time | 当前任务各阶段/合计时长 | 受测试机器及图片影响，不宜跨机比较 |

## 6. 10 分钟展示建议

1. 0:00–1:20：题目、目标、完整算法流程。
2. 1:20–2:40：介绍 Baseline vs APDrawingGAN、前处理必要性。
3. 2:40–4:30：展示真实上传与 Job `queued → running → completed`。
4. 4:30–6:00：展示 Original → Prepared → Aligned → Portrait → Robot Vector。
5. 6:00–7:15：对比 Aligned 与 Portrait，以及 Portrait 与 Robot Vector。
6. 7:15–8:35：SVG Path Pen-down / Pen-up 动画和路径优化指标。
7. 8:35–10:00：实验/局限/总结，并预留问答时间。

演示前务必完成一次完整流程；准备预生成的 SVG、PNG、录屏作为 WSL/CUDA 故障备用，不要现场排查驱动。

## 7. 本地运行 / 冻结

```powershell
# 项目根目录，已有 .venv 与 WSL checkpoint 环境
python -m pytest -q
.\scripts\start_web.ps1

# 另一个终端
cd frontend
npm ci
npm run typecheck
npm run build
npm run dev
```

进入 `http://127.0.0.1:5173`；主方法 APDrawingGAN + Centerline + 512；确保 PNG/SVG/manifest 可下载。

冻结版本后：提交特定改动到独立 Git 分支，合并 PR；PPT 与报告统一使用该版本的截图/指标/路径，不混用旧版本数据。

## 8. 可复现的画质检查（不影响生产）

仅针对已生成的 `apdrawinggan.png` 执行：

```powershell
python scripts/compare_vector_presets.py --input "outputs/web_jobs/<job_id>/apdrawinggan.png"
```

会在 `outputs/vector_comparison/` 下产生 stable / clean_experimental 两套 SVG、PNG 和 metrics。**实验参数可能损害眼睛/鼻梁/发丝细节，最终是否采用必须结合真人视觉评估；本版没有将其设为默认。**

## 9. 目录与接口

- API: `GET /api/health`, `GET /api/methods`, `POST /api/jobs`, `GET /api/jobs/{id}`, `GET /api/jobs/{id}/manifest`, `GET /api/jobs/{id}/files/{filename}`。
- 前端：`frontend/src/App.vue`, `components/JobResults.vue`, `ResultComparison.vue`, `DrawingSimulation.vue`。
- 后端：`src/robot_portrait/web/`；核心算法：`pipeline.py`, `centerline.py`, `vectorize.py`, `generators/`。
- 本地 checkpoint、数据集、WSL GPU 环境不会包含在源码 ZIP 中；不能声称下载源码即可零配置运行完整 GAN。
