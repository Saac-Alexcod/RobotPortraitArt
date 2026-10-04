# 输入照片

将自己的单张人脸照片放在此目录，例如 `face.jpg` 或 `face.png`。命令中的扩展名必须与实际文件名一致；当前工作区示例文件是 `face.png`，因此应运行：

```powershell
python scripts/prepare_image.py --input data/input/face.png --output outputs/face_prepared.png --size 512
```

仓库中的 `demo_face.png` 是流程演示图，不代表真实实验数据。
