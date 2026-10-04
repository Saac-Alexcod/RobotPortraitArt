# APDrawingGAN 输入暂存目录

将已经按官方 `third_party/APDrawingGAN/preprocess/readme.md` 完成对齐的 512×512 人脸图放在这里。

使用 `--use_local` 推理时，还需要为每张图准备同名文件：

- `third_party/APDrawingGAN/dataset/landmark/ALL/<stem>.txt`：5 点人脸关键点；
- `third_party/APDrawingGAN/dataset/mask/ALL/<stem>.png`：背景 mask。

例如：

```text
data/apdrawing_preprocessed/face01.png
third_party/APDrawingGAN/dataset/landmark/ALL/face01.txt
third_party/APDrawingGAN/dataset/mask/ALL/face01.png
```

`prepare_image.py` 只负责通用 EXIF 旋转、方形裁剪和尺寸统一，不会自动生成这两个 APDrawingGAN 专用文件。请不要把未经对齐或缺少关键点/mask 的普通照片直接当作 APDrawingGAN 输入。
