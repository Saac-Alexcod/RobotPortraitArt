# 本地输入照片

这里用于放置你自己的测试照片或合法获得的授权照片，例如 `data/input/face.jpg`。该目录中的个人输入默认由 `.gitignore` 忽略，不会自动提交到 Git。

无需自己寻找照片时，可以直接使用公开的项目示例：

```text
data/examples/portrait_sample.jpg
```

在 Web 界面手动选择该文件，或在项目根目录执行：

```powershell
python scripts/run_pipeline.py `
  --input data/examples/portrait_sample.jpg `
  --output-dir outputs/sample_baseline `
  --method baseline `
  --config configs/centerline.yaml `
  --size 512
```

`data/input/demo_face.png` 是简笔画测试素材，不是自然人脸照片，不能用于证明 YuNet 或 APDrawingGAN 的真实人脸处理性能。
