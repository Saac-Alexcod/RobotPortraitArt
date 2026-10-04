# 实验记录

本文件用于记录课程项目的真实实验配置、结果和结论。所有数值必须由实际程序运行得到，不得预先编造。

## 当前资源状态

- `data/input/`：请放入自己的测试照片；仓库不包含用户照片。
- `data/CelebA/`：当前未提供 CelebA 原始数据。
- `data/apdrawing_preprocessed/`：当前未提供 APDrawingGAN 预处理数据。
- `third_party/APDrawingGAN/`：当前未提供官方仓库和模型权重。
- 如上述资源为空，只能记录代码 smoke test 和演示流程，不能填写 E1/E2 的数值。

## 数据集显现记录

使用 `scripts/visualize_dataset.py` 生成联系图后，在此记录目录、样本数和输出文件。联系图用于检查数据放置，不替代评价指标。

## E0——OpenCV 线稿基线

- 输入数据：已用 `data/CelebA/Img/img_align_celeba/000001.jpg` 做流程 smoke test；正式实验仍应使用 `data/input/<sample>.jpg`（先由 `scripts/prepare_image.py` 输出 `outputs/<sample>_prepared.png`）
- 配置文件：`configs/default.yaml`
- 输出栅格图：`outputs/celeba_000001_line.png`（smoke test）
- 输出 SVG：`outputs/celeba_000001_line.svg`（smoke test）
- 对比图：`outputs/celeba_000001_comparison.png`
- SSIM：未计算；当前没有该样本的独立参考线稿
- Edge-F1：未计算；当前没有该样本的独立参考线稿
- LPIPS：可选依赖，待真实运行后填写
- 人脸局部人工评分：待填写（eyes / eyebrows / nose / mouth / contour / hair）
- 实验现象与分析：待填写；不要用单一像素误差判断风格

## E1——APDrawingGAN 预训练模型

- 使用仓库：`yiranran/APDrawingGAN`
- 模型检查点：
- 数据预处理方式：
- 输出结果：官方 `results/` 或项目记录的 `outputs/apdrawinggan/`（以实际仓库配置为准）
- SSIM：待真实运行后填写
- Edge-F1：待真实运行后填写
- LPIPS：可选依赖，待真实运行后填写
- 人脸局部人工评分：待填写
- 实验现象与分析：待填写；记录身份结构和线条风格的优缺点

## E2——改进后的神经网络方案

可以选择以下一种方式：

- 对 APDrawingGAN 的预处理或参数进行消融/改进；或
- 使用 APDrawingGAN++。

需要准确记录修改内容，并保持其他评价条件一致。

- 改进内容：待选择并明确记录（例如更大方形裁剪边距、局部预处理消融或 APDrawingGAN++）
- 输入数据：与 E1 完全相同
- 模型/检查点：待填写真实路径
- 输出结果：`outputs/apdrawinggan_e2/`
- SSIM：待真实运行后填写
- Edge-F1：待真实运行后填写
- LPIPS：可选依赖，待真实运行后填写
- 人脸局部人工评分：待填写
- 与 E1 的对比：待填写，保持评价脚本和参考图一致
- 实验结论：待填写

## 人工局部评分说明

对以下部位分别按 1～5 分评价：

- 眼睛；
- 眉毛；
- 鼻子；
- 嘴巴；
- 脸部轮廓；
- 头发。

评分时重点观察局部几何结构是否保持、线条是否自然，以及生成结果是否仍能反映输入人物的主要肖像特征。
