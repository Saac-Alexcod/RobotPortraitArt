# 卡通风格与课程目标验收

保持 Windows 启动 FastAPI/Vue、Windows 完成预处理和矢量化、WSL/CUDA 执行 GAN 推理。新增方法 `cartoon` 使用课题图片列出的 **GANs N’ Roses**；原有 APDrawingGAN 和 Baseline 保留。前端布局、路径规划和逐笔模拟算法未重写。

## 启动

本机模型已经安装。重启后端以加载新增代码，刷新前端后选择「GANs N’ Roses · 卡通」。卡通模型固定 256×256；APDrawingGAN 固定 512×512。建议采用 Centerline 绘制。

```powershell
.\scripts\start_web.ps1
# 另开终端（项目根目录）
cd frontend
npm run dev
```

其他机器首次配置：

```powershell
# WSL PyTorch/CUDA 环境尚未安装时，可同时初始化
.\scripts\start_web.ps1 -SetupAPD -SetupCartoon
# 已有 APD WSL 环境，只安装卡通模型
.\scripts\start_web.ps1 -SetupCartoon
```

脚本使用现有 WSL 环境发现逻辑，不硬编码用户名。`-Distro`、`-APDPython` 的覆盖规则不变。普通启动不会自动下载卡通权重；没有模型时只禁用该方法。安装脚本从 Windows 网络下载约 302 MiB 权重到 `third_party/GANsNRoses/`，无需额外安装训练或视频依赖。`-CheckOnly` 不安装、不启动服务。

源码：[mchong6/GANsNRoses](https://github.com/mchong6/GANsNRoses)，固定 revision `59bf4680c9c700890cdf53ffac86a6f7da8a9153`。下载的最小推理源码保留 MIT LICENSE；权重来自该 revision 的 `util.py` 所指向的官方 `GNR_checkpoint.pt`。SHA256：`36eb8254155dffb12aed780d1ea479a5bb852e2087dea72e032e2290705a0cfb`。安装目录记录 `provenance.json`，源码与权重均由脚本获取，不把大文件提交到 Git。推理使用 `weights_only=True` 并严格加载 `G_A2B_ema`。

## 任务与产物

| 课题要求 | 实现与可检查产物 |
|---|---|
| 画家风格肖像 | 原有 APDrawingGAN：肖像 PNG |
| 扩展动漫/卡通 | GANs N’ Roses：彩色卡通 PNG，固定 seed=42 可复现同一风格 |
| 矢量描述、绘制显示 | 卡通轮廓 PNG → 现有 Contour/Centerline → SVG → 逐笔模拟 |
| 风格图相似度 | Manifest 和前端「技术数据」：原图→风格图的 SSIM、容差 Edge-F1 |
| 矢量图相似度 | 原图→SVG 回栅格图的 SSIM、容差 Edge-F1；另测绘制线条→SVG 的细节保留 |
| 逼真度/风格质量 | 实测对照画廊 + 人工评价表；不可从结构指标直接推导 |
| 端到端机器人思考 | 上传→裁剪/对齐→GAN→轮廓→SVG 路径优化→模拟；实体机械臂仍需坐标标定、笔压/抬笔控制、限位和硬件验收 |

卡通保持彩色栅格输出；绘制采用单笔黑色轮廓，不包含彩色填充。不同模型使用不同的人脸预处理；分数不是在完全相同裁剪上的严谨模型排名。

## 实测与评分方式

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_course.py
# 自己的数据集，单独目录保存本次结果
.\.venv\Scripts\python.exe scripts/evaluate_course.py --input-dir data/input --output-dir outputs/my_evaluation
```

默认使用 `data/examples/` 四张合成演示人像，运行 Baseline、APDrawingGAN 和卡通三种方法（12 次）。输出位于 `outputs/course_evaluation/`：

- `report.html`：原图裁剪/对齐、风格图、矢量图逐例对照；失败案例也保留。
- `scores.csv` / `results.json`：实际测量的结构指标、路径数和耗时。
- `human_review.csv`：人工评分空表，不伪造评价结果。
- 各任务目录：PNG、SVG、Manifest，包括全部测量参数。

SSIM 使用灰度，范围约 -1～1；Edge-F1 为 0～1，越高表示对应结构越接近。均统一到 256×256，轮廓允许 2 px 偏差。照片和彩色卡通使用 GaussianBlur+Canny(50,150)，线稿/矢量采用阈值 230 后骨架化。空参考轮廓记缺失，预测为空而参考非空记 0。评估使用同一生成器的真实输入几何；APD 使用对齐原图。保留原始分数，不把 SSIM×100 宣称为课程得分。

建议人工评价每项 1～5：身份/五官相似、风格自然度、矢量肖像相似、无伪影程度。1=明显不符合，3=可辨认但有明显缺陷，5=结构保留良好且风格自然；至少三名独立评价者逐样例填写，报告均值、分歧和失败比例。若进行盲评，需另行隐藏方法名称并随机呈现样例。这是本项目建议的评价方案，图片未提供官方分值、权重或及格阈值，最终课程成绩由教师确定。

上游明确说明训练集主要为女性，男性域效果较差。本机示例也出现男性转为女性动漫形象、眼镜丢失及发型改变；卡通风格转换完成，但不能据此宣称身份相似性达标。未进行新训练、未使用真实机器人验证，四张演示图片不构成总体质量结论。不要把预训练推理称为自己完成 GAN 训练。
