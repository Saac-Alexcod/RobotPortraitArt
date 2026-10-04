# 机器人肖像绘制艺术风格学习（L1）

这是一个保留现有工程结构的课程项目，采用“可运行 OpenCV 基线 + APDrawingGAN 适配器 + SVG 矢量化 + 自动评价”的路线。主工程不修改第三方模型代码；APDrawingGAN 放在 `third_party/`，并使用独立的 legacy 环境。

主工程支持现代 Python，并且在没有 GPU、APDrawingGAN 权重或 CelebA 的情况下仍可运行 baseline、SVG 和结构指标流程。神经网络结果只有在用户准备真实仓库、权重和输入数据后才会生成，文档和报告不会填入虚构数值。

## 当前项目状态

- 已实现：OpenCV 线稿基线、SVG 轮廓矢量化、SVG 回栅格接口、SSIM、Edge-F1、可选 LPIPS。
- 已补齐：EXIF 旋转/人脸检测/方形裁剪预处理、缺失结果自动跳过的对比联系图、批量评价、数据集图片联系图。
- 已通过：`pytest` 核心测试（6 项）和全部新增脚本的语法检查。
- 测试缓存统一配置在 `.cache/`，主目录不会再生成 `pytest-cache-files-*` 临时目录。
- 已加入：`third_party/APDrawingGAN/` 官方代码副本（保留上游 LICENSE），以及本地 CelebA 图片数据。
- 尚未完成：APDrawingGAN 预训练 checkpoint、与其格式匹配的对齐图/关键点/背景 mask，以及真实 E1/E2 实验；不会编造结果。

## 目录与数据放置

```text
robot_portrait_art_project/
├─ data/
│  ├─ input/                    # 私人输入照片（不提交；face.jpg/png 等）
│  ├─ reference_style/          # 与测试照片同名的参考线稿
│  ├─ CelebA/                   # 原始 CelebA，不提交到 Git
│  └─ apdrawing_preprocessed/   # APDrawingGAN 对齐/关键点/局部预处理结果
├─ outputs/                     # PNG、SVG、联系图、JSON、metrics.csv、summary.json
├─ configs/default.yaml
├─ scripts/
│  ├─ prepare_image.py          # EXIF、人脸检测、居中方形裁剪
│  ├─ run_baseline.py
│  ├─ compare_grid.py
│  ├─ batch_evaluate.py
│  ├─ rasterize_svg.py         # SVG 回栅格 CLI（优先 cairosvg，支持 OpenCV fallback）
│  ├─ visualize_dataset.py     # 数据集联系图/显现
│  ├─ evaluate.py
│  └─ run_apdrawinggan_adapter.py
├─ src/robot_portrait/          # 线稿、SVG、指标
├─ third_party/APDrawingGAN/    # 已加入的官方第三方代码副本（无 checkpoint）
├─ environment_apdrawinggan.yml # 独立 legacy 环境参考
└─ tests/
```

自己的照片放到 `data/input/`；当前目录中的 `demo_face.png` 只是用于验证流程的合成演示图。CelebA 图片位于 `data/CelebA/Img/img_align_celeba/`（注意 `Img` 这一层目录）。APDrawingGAN 的对齐输入放到 `data/apdrawing_preprocessed/`；启用其 `--use_local` 推理还需要每张图同名的 5 点关键点 `.txt` 和背景 mask `.png`，默认分别放在 `third_party/APDrawingGAN/dataset/landmark/ALL/` 与 `third_party/APDrawingGAN/dataset/mask/ALL/`。模型权重应放在官方工程要求的 `third_party/APDrawingGAN/checkpoints/formal_author/`，不放入主工程 requirements。

## 安装

主工程（CPU baseline）只安装必要依赖：

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

如果 PowerShell 提示符同时出现 `(.venv) (base)`，Conda 的 `python` 可能仍优先于项目虚拟环境。先检查：

```powershell
python -c "import sys; print(sys.executable)"
python -m pip show cairosvg
```

正确解释器应为本项目的 `.venv\Scripts\python.exe`。若输出仍是 Anaconda，可以执行 `conda deactivate` 后重新激活 `.venv`，或直接使用不受 PATH 影响的命令：

```powershell
.\.venv\Scripts\python.exe scripts\compare_grid.py --original outputs/face_prepared.png --baseline outputs/face_line.png --svg outputs/face_line.svg --output outputs/comparison.png
```

可选依赖单独安装，不会污染 baseline 环境：

```bash
pip install -r requirements-optional.txt
```

若只需要 CPU baseline，不要安装可选文件；若启用 LPIPS，请先按机器选择匹配的 CPU/GPU PyTorch wheel，再安装 LPIPS。SVG 回栅格优先使用 CairoSVG；Windows 缺少 Cairo 原生 DLL 时，项目会自动对自己生成的简单 M/L/Z SVG 使用 OpenCV fallback。

Windows 出现 `no library called "cairo-2" was found` 时，说明 Python 包已安装但系统缺少原生 Cairo DLL。本项目会优先尝试 CairoSVG；若该 DLL 不可用，则自动使用 OpenCV 回栅格本项目生成的 M/L/Z SVG，并在终端标明 `opencv-fallback`。因此无需为了完成课程 baseline 强制安装系统 GTK/Cairo；任意第三方复杂 SVG 仍应安装完整 Cairo runtime。

检查主工程：

```bash
pytest -q
```

## 运行前检查清单

在执行完整流程前逐项确认：

- 当前目录是项目根目录：`Get-Location`；
- 当前解释器是 `.venv\Scripts\python.exe`：`python -c "import sys; print(sys.executable)"`；
- 输入图片真实存在，且扩展名完全一致：`Get-ChildItem data/input`；
- CPU baseline 只需 `requirements.txt`，LPIPS/CairoSVG 才需要 `requirements-optional.txt`；
- `compare_grid.py` 是可视化，不是定量评价；
- 正式评价前必须放入真实参考线稿，例如 `data/reference_style/face.png`；
- APDrawingGAN 推理前必须另外准备 checkpoint、512×512 对齐输入、关键点和背景 mask；
- 所有报告数值必须来自实际命令输出，演示图和 smoke test 不得写成正式实验结果。

## 单张图片完整流程

运行前必须先将自己的照片放入 `data/input/`，并在命令中使用实际扩展名（例如 `face.jpg`、`face.jpeg` 或 `face.png`）。对应参考线稿再放入 `data/reference_style/` 并保持相同 stem；如果只是检查程序，可以直接使用仓库中的 `demo_face.png`，但它是合成演示图，不是实验数据。

```bash
python scripts/prepare_image.py --input data/input/face.png --output outputs/face_prepared.png --size 512
python scripts/run_baseline.py --input outputs/face_prepared.png --output outputs/face_line.png --svg outputs/face_line.svg
python scripts/compare_grid.py --original outputs/face_prepared.png --baseline outputs/face_line.png --svg outputs/face_line.svg --output outputs/comparison.png
```

`compare_grid.py` 只负责视觉对比，不计算指标，也不会把输入照片或 baseline 自身当作参考线稿。艺术结果评价只有在真实的 `data/reference_style/face.png` 存在时才能运行：

```powershell
python scripts/evaluate.py --pred outputs/face_line.png --ref data/reference_style/face.png --json_out outputs/face_metrics.json
```

如果暂时没有人工参考线稿，可以先评价 SVG 矢量化保真度（这是“线稿 PNG 与 SVG 回栅格图的一致性”，不是艺术风格质量）：

```powershell
python scripts/rasterize_svg.py --input outputs/face_line.svg --output outputs/face_line_raster.png --width 512
python scripts/evaluate.py --pred outputs/face_line_raster.png --ref outputs/face_line.png --json_out outputs/face_vector_fidelity_metrics.json
```

`prepare_image.py` 和 `run_baseline.py` 默认分别写出同名 JSON 配置/处理记录；可用 `--metadata` 指定路径。两者都接受 `--seed`（默认 42），便于报告复现。一次只生成 PNG + SVG 的最小命令是：

```bash
python scripts/run_baseline.py --input data/input/demo_face.png --output outputs/demo.png --svg outputs/demo.svg
```

使用演示图时，将上述命令中的输入改为 `data/input/demo_face.png`；评价命令只有在 `data/reference_style/face.png` 存在时才能运行。

没有检测到人脸时，预处理会输出中文 warning 并使用中心方形裁剪，不会直接崩溃。若 CairoSVG 或 Windows 原生 Cairo 不可用，回栅格脚本会明确显示 `opencv-fallback`；该 fallback 只保证本项目 `raster_to_svg` 生成的 M/L/Z 路径，不承诺支持任意复杂 SVG。

## 数据集显现

先确认数据确实放在目标目录，再生成联系图：

```bash
python scripts/visualize_dataset.py --input-dir data/input --output outputs/input_contact_sheet.png
python scripts/visualize_dataset.py --input-dir data/CelebA/Img/img_align_celeba --output outputs/celeba_contact_sheet.png --max-images 100
```

脚本只读取常见图片扩展名，联系图中的文件名用于核对样本，不代表实验指标。
如果 CelebA 尚未下载或目录为空，脚本会输出中文 warning 并安全结束；准备好数据后重新运行即可。若你的解压工具生成的是 `data/CelebA/img_align_celeba/`，也可以直接使用该路径；关键是 `--input-dir` 必须指向实际包含 JPG/PNG 文件的目录。

## 批量评价

参考图和预测图按文件名 stem 对齐。例如 `data/reference_style/face01.png` 对应 `outputs/baseline/face01.png`：

PowerShell 可直接使用一行命令：

```powershell
python scripts/batch_evaluate.py --reference-dir data/reference_style --method baseline=outputs/baseline --method apdrawinggan=outputs/apdrawinggan --output-csv outputs/metrics.csv --summary-json outputs/summary.json --svg-dir outputs/svg
```

Bash 也可以拆行：

```bash
python scripts/batch_evaluate.py \
  --reference-dir data/reference_style \
  --method baseline=outputs/baseline \
  --method apdrawinggan=outputs/apdrawinggan \
  --output-csv outputs/metrics.csv \
  --summary-json outputs/summary.json \
  --svg-dir outputs/svg
```

缺少某个方法的图片时，CSV 会保留样本行并写明“缺少预测图，尚未运行”；人工局部评分（eyes、eyebrows、nose、mouth、contour、hair）需要在真实观察后填写。绝不预填 SSIM、Edge-F1、LPIPS 或 APDrawingGAN 结果。

SVG 回栅格可单独执行：

```bash
python scripts/rasterize_svg.py --input outputs/face_line.svg --output outputs/face_line_raster.png --width 512
```

Windows 下若 CairoSVG 已安装在 `.venv`，但普通 `python` 仍报告未安装，请显式运行：

```powershell
.\.venv\Scripts\python.exe scripts\rasterize_svg.py --input outputs/face_line.svg --output outputs/face_line_raster.png --width 512
```

如果需要确认实际后端，可加 `--verbose`；输出中的 `backend` 为 `cairosvg` 或 `opencv-fallback`。`--require-svg` 表示 SVG 必须成功回栅格（CairoSVG 或 fallback 任一成功都算成功），只有两种后端都失败时才退出报错。

## APDrawingGAN

```bash
git clone https://github.com/yiranran/APDrawingGAN third_party/APDrawingGAN
conda env create -f environment_apdrawinggan.yml
python scripts/run_apdrawinggan_adapter.py \
  --repo third_party/APDrawingGAN \
  --input_dir data/apdrawing_preprocessed \
  --experiment formal_author --epoch 300 \
  --python <APDrawingGAN旧环境中的python>
```

适配器只负责调用官方 `test.py`，不会把 legacy 依赖混入主环境。运行前必须准备官方仓库、权重和符合其格式的预处理数据。`environment_apdrawinggan.yml` 是隔离环境参考；旧仓库实际依赖若有变化，应以官方 README 为准。

适配器不会下载或伪造权重；缺少 `third_party/APDrawingGAN/test.py`、checkpoint 或输入数据时会直接给出可操作的错误。`--log-json`（默认 `outputs/apdrawinggan_adapter.json`）记录调用参数和 seed，但官方旧脚本内部是否使用随机数仍以其源码为准。

当前工作区已将相邻目录中的 APDrawingGAN 源码复制到 `third_party/APDrawingGAN/`，官方 LICENSE 和示例文件均保留；这一步只加入代码，不包含预训练权重。官方预处理说明要求 512×512 对齐人脸、5 点关键点和背景 mask。`prepare_image.py` 只负责通用 EXIF/人脸裁剪与缩放，不会自动生成 APDrawingGAN 所需的关键点或背景 mask，因此不能直接把它的输出当作完整 APDrawingGAN 输入。

用现有 CelebA 样本检查 E0 baseline（只作流程演示，不是单独的测试集统计）：

```powershell
python scripts/prepare_image.py --input data/CelebA/Img/img_align_celeba/000001.jpg --output outputs/celeba_000001_prepared.png --size 512
python scripts/run_baseline.py --input outputs/celeba_000001_prepared.png --output outputs/celeba_000001_line.png --svg outputs/celeba_000001_line.svg
python scripts/compare_grid.py --original outputs/celeba_000001_prepared.png --baseline outputs/celeba_000001_line.png --svg outputs/celeba_000001_line.svg --output outputs/celeba_000001_comparison.png
```

## 评价原则与实验记录

SSIM、Edge-F1 越高越好，LPIPS 越低越好；SVG 还记录 path 数量和文件字节数。E0/E1/E2 的结果请写入 `EXPERIMENTS.md` 和 `REPORT_DATA_TEMPLATE.csv`，所有数值必须来自实际运行。当前项目包含 CelebA 数据与可运行的 E0 baseline；正式艺术质量指标仍需要独立参考线稿，E1/E2 仍需要 APDrawingGAN 权重及其专用预处理输入。

评价注意事项：

- SSIM 反映整体结构相似度，不等价于艺术风格相似度；
- Edge-F1 反映边缘/线条重合程度；
- LPIPS 是可选的感知距离，越低越好；未安装依赖时 CSV 留空；
- `svg_paths` 和 `svg_bytes` 描述矢量复杂度，不代表质量高低；
- eyes、eyebrows、nose、mouth、contour、hair 六项必须人工按 1～5 分填写。

每次 baseline 运行会默认写出同名 JSON，例如 `outputs/face_line.json`；其中包含 seed、输入、配置和输出路径。批量评价的 `summary.json` 也会记录 seed 和完成行数。

## 常见问题与注意事项

### 输入文件名和路径

- `--input data/input/face.jpg` 只会查找精确的 `face.jpg`；如果实际文件是 `face.png`、`face.jpeg` 或 `face.JPG`，命令必须写实际名称。
- 当前工作区示例输入是 `data/input/face.png`，不是 `face.jpg`。
- Windows 相对路径相对于当前终端目录解析；请先确认 `Get-Location` 位于项目根目录。
- 输入不存在时，`prepare_image.py` 会列出 `data/input/` 中检测到的图片候选，不会自动把另一张图片当作输入。

### Conda 与 `.venv`

- 提示符同时出现 `(.venv) (base)` 时，`python` 可能仍指向 Anaconda。
- 用下面命令确认解释器和包属于同一环境：

```powershell
python -c "import sys; print(sys.executable)"
python -m pip show cairosvg
```

- 若路径不是项目的 `.venv\Scripts\python.exe`，执行 `conda deactivate` 后重新激活 `.venv`，或始终使用 `.\.venv\Scripts\python.exe` 的绝对路径。

### 对比图、回栅格与评价的区别

- `compare_grid.py` 只生成 Original / Baseline / APDrawingGAN（若提供）/ Vector Render 的视觉对比图，不计算 SSIM、Edge-F1 或 LPIPS。
- `rasterize_svg.py` 的 `backend=cairosvg` 是标准 SVG 渲染；`backend=opencv-fallback` 是本项目专用的简化渲染，仅支持本项目生成的 M/L/Z 路径。
- `evaluate.py` 的艺术质量评价必须使用独立的真实参考线稿，不能把输入照片、baseline 输出或 SVG 回栅格输出复制到 `data/reference_style/` 冒充参考答案。
- 没有参考线稿时，只能运行 PNG 与 SVG 回栅格之间的“矢量化保真度”评价；该结果不代表艺术风格质量。
- LPIPS 第一次运行可能下载 AlexNet 权重（约 233 MB），需要网络；下载失败时 LPIPS 留空，不影响 SSIM 和 Edge-F1。

### CelebA 路径

- 本项目当前图片目录是 `data/CelebA/Img/img_align_celeba/`，不是 `data/CelebA/img_align_celeba/`。
- `visualize_dataset.py` 遇到不存在的路径会搜索并提示可能的 `img_align_celeba` 实际目录。

### APDrawingGAN 资源

- `third_party/APDrawingGAN/` 只包含官方代码和 LICENSE，不等于已经下载 checkpoint。
- APDrawingGAN `--use_local` 需要 512×512 对齐图、5 点关键点 `.txt` 和背景 mask `.png`；`prepare_image.py` 不会生成这些专用文件。
- 缺少 checkpoint、预处理输入或关键点/mask 时，不能声称 E1/E2 已完成，也不能填写虚构指标。

- 人脸检测依赖 OpenCV Haar cascade。若当前 OpenCV 构建缺少该接口、cascade 文件或检测失败，脚本会警告并退回中心方形裁剪。
- 没有 `cairosvg` 或系统 Cairo DLL 时，PNG/SVG 基线仍可运行；项目生成的 SVG 会使用 OpenCV fallback，复杂第三方 SVG 可能仍需完整 Cairo runtime。
- 没有参考线稿时不能计算有意义的 SSIM、Edge-F1 或 LPIPS；程序不会用演示图结果冒充真实实验。

参考工程：APDrawingGAN、APDrawingGAN++、U-GAT-IT、GANs N' Roses、CelebA。使用时请遵守各自许可证和数据协议。
