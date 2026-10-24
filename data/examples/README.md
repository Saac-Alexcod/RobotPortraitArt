# 公开示例人像 / Public sample portraits

本目录提供 **4 张由生成式图像工具制作的合成人像**，用于本地功能演示。它们不是 CelebA 原始数据，也不是拍摄的真人照片。无需另外下载大型测试集。

| 文件 | 特征 | 建议用途 |
|---|---|---|
| `portrait_sample.jpg` | 正面人像，短发，无眼镜 | 首次运行推荐；已验证 Baseline → Centerline → SVG |
| `portrait_long_hair.jpg` | 正面人像，长发 | 测试发丝及黑色区域的矢量化效果 |
| `portrait_glasses.jpg` | 正面人像，佩戴眼镜 | 测试眼镜轮廓和脸部细节的保留 |
| `portrait_profile_challenge.jpg` | 明显侧转人像 | **挑战样例**；可观察姿态对检测/对齐的影响，GAN 自动模式可能拒绝处理 |

每张图均为 768×768 JPEG，小于 100 KB。示例用于代码功能体验和定性观察，**不是官方人脸检测、GAN 或画质指标的基准测试集**。新增样例经过文件解码与 OpenCV Baseline 流水线检查；这不意味着每张样例已经通过 YuNet + PPHumanSeg + APDrawingGAN 的完整模型推理。

## 使用方式

Web 工作台现在提供四张可点击的示例缩略图，选择后会自动填入上传区域。它们作为静态资源同步放在 `frontend/public/examples/`，点击后仍通过正常文件上传接口创建任务。也可以手动选择 `.jpg` 文件；或在项目根目录运行：

```powershell
python scripts/run_pipeline.py `
  --input data/examples/portrait_long_hair.jpg `
  --output-dir outputs/example_long_hair `
  --method baseline `
  --config configs/centerline.yaml `
  --size 512
```

生成的 PNG、SVG 和 Manifest 均保存在 `outputs/` 中，默认不提交 Git。若使用 APDrawingGAN，请先自行配置模型权重和预处理依赖，并以实际运行结果为准。原有 `data/input/demo_face.png` 是绘制的图形测试图，并非用于 GAN 自动人脸检测的照片样例。

请勿将随机下载的真实人物照片或 CelebA 图片加入公开仓库，除非已确认肖像及再分发授权。
