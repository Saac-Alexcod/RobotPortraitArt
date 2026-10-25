# 课程任务验收记录 · 2026-10-25

本次实测为 Windows 主程序 + WSL Ubuntu-22.04 / PyTorch 2.5.1+cu121 / RTX 4060 Laptop GPU。使用仓库 `data/examples/` 的四张合成人像，Baseline、APDrawingGAN、GANs N’ Roses 三种方法均实际运行，**12/12 成功**。全部使用 Centerline、seed=42。卡通原生分辨率 256×256，其余 512×512；指标统一在 256×256 上计算。

## 测量结果

以下为每种方法四张图的平均值，均来自 `outputs/course_evaluation/results.json`，不是人工质量等级或课程总分。

| 方法 | 原图→风格图 SSIM | 原图→风格图 Edge-F1 | 原图→矢量图 SSIM | 原图→矢量图 Edge-F1 | 绘制线条→SVG Edge-F1 | Pipeline 秒 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.5040 | 0.6868 | 0.5132 | 0.7282 | 0.9688 | 1.47 |
| APDrawingGAN | 0.5523 | 0.5498 | 0.5024 | 0.5587 | 0.8556 | 10.64 |
| GANs N’ Roses | 0.4045 | 0.4562 | 0.3571 | 0.4606 | 0.9973 | 6.88 |

照片原图按该生成器实际裁剪/对齐几何进行比较；不同模型的输入区域并不完全相同，因此这些数值适合定位误差，不适合宣称模型排名。Pipeline 耗时不含排队、状态预检和浏览器加载。

**卡通结果的解释：** 绘制线条→SVG 保留率高，说明现有矢量化能保留输入轮廓；原图→卡通结构指标较低，说明主要变化发生在 GAN 风格迁移阶段。两张男性示例失去了眼镜或男性外观、发型明显变化；女性正面和侧脸示例保留了部分姿态，但眼睛被夸大。APDrawingGAN 正面结果保留更多原始五官；侧脸示例仍可见嘴部黑块、边缘噪声等问题。不能用“成功生成”替代身份相似性或艺术质量验收。

## 可查看的交付物

- [完整对照报告](../outputs/course_evaluation/report.html)
- [逐样例评分 CSV](../outputs/course_evaluation/scores.csv)
- [机器可读结果](../outputs/course_evaluation/results.json)
- [人工评价表](../outputs/course_evaluation/human_review.csv)
- [四张图对照拼图](../outputs/course_evaluation/comparison.jpg)
- [前端指标截图](../outputs/course_evaluation/browser-metrics.png)
- [前端绘制截图](../outputs/course_evaluation/browser-drawing.png)

以上运行产物保留于本机 `outputs/`，按仓库规则不提交 Git；可用 [复现命令](CARTOON_SETUP.md) 重新生成核心报告。

## 软件验证

- 卡通适配器、尺寸与可用性接口、彩色图与矢量线条分离、空图评分、原有 Pipeline/API/Worker/启动器/APD 自动流程：**71 项测试通过**。
- `npm run build`：Vue TypeScript 检查及生产构建通过。
- 真实 HTTP 上传 → Windows 后端 → WSL CUDA 推理 → SVG → Manifest → PNG/SVG 下载通过。
- Edge 无头浏览器验证：卡通选项可选，尺寸自动设为 256，示例图片提交成功，技术数据可见，无 JavaScript 页面错误；绘制模拟可加载实际 SVG，播放后进度增长、暂停操作通过。

## 与评分要求的边界

软件已覆盖课题的艺术肖像、卡通扩展、矢量化、绘制显示和两组结构相似度测量。生成模型使用公开预训练权重，本次没有新训练。卡通模型在男性输入上的身份保留不足，不能宣称全部人像都达到相似度要求。

图片没有给出分值权重、及格阈值或目标画作；**最终艺术质量与正式成绩尚不能确定**。人工评价表保持空白，供评价者按身份相似、风格自然、矢量相似和伪影四项 1～5 分填写；建议至少三名独立评价者。实体机器人的坐标标定、运动控制、笔压与硬件验证不在已完成的软件模拟结果之内。
