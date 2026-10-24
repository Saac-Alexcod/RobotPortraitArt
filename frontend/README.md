# RobotPortraitArt · Final Course Demo (2026-10-24)

Vue 3 + TypeScript + Vite frontend for the RobotPortraitArt FastAPI service.

## Requirements

- Node.js `^20.19.0` or `>=22.12.0`
- npm 10+
- RobotPortraitArt Python environment with the Web API dependencies installed

## Install

```powershell
cd frontend
npm ci
```

## Validate

```powershell
npm run typecheck
npm run build
```

## Develop

Start the FastAPI backend from the repository root:

```powershell
python -m uvicorn `
  robot_portrait.web.app:app `
  --app-dir src `
  --host 127.0.0.1 `
  --port 8000
```

If APDrawingGAN is configured through the WSL backend, set the project-specific
environment variables required by the backend before starting Uvicorn.

Then start Vite in another terminal:

```powershell
cd frontend
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

Vite proxies `/api/*` to `http://127.0.0.1:8000`.

## Implemented Web Demo scope

- live FastAPI health and generator availability
- image selection, drag-and-drop, validation, and local preview
- Baseline / APDrawingGAN generator selection based on backend availability
- Contour / Centerline vector mode selection
- real `POST /api/jobs` job creation
- automatic queued / running / completed / failed polling
- Original / Prepared / Portrait / Robot Vector result gallery
- Manifest-backed stroke, pen-down, pen-up, vectorization, and total-time metrics
- PNG / SVG / Manifest downloads
- interactive A/B result comparison
- real SVG path robot drawing animation with Pen-down / Pen-up visualization

The frontend intentionally reuses the existing FastAPI API and does not
reimplement portrait generation or vectorization logic in the browser.


## Presentation conventions

- Default selected generator: APDrawingGAN when reported available; otherwise fall back to Baseline.
- Pipeline cards describe the real processing stages, not release history.
- 2-opt ratio comes from server Manifest as a fraction; the UI displays it as a percentage.
- The Robot Drawing panel plays the *software-generated* SVG path order; it does not command real robot hardware.
- On errors, check `/api/health`, `/api/methods`, and the browser Network tab. No account/database is required.

## Defense UI Results update (UI RC2)

- Two large, clickable previews highlight GAN Portrait and the real Robot Vector output; press Escape or the close button to exit the preview.
- Original, Prepared, and optional Aligned Input stay visible in a compact processing history.
- Manifest-based metrics include a Greedy vs 2-opt distance comparison when both source numbers are present. These are image-pixel distances, not measured robot motion.
- A/B comparison offers quick presets for aligned-input/portrait and portrait/vector, while retaining manual selection.
- No change was made to FastAPI, the GAN, vectorization parameters, or the SVG output.

## Studio layout (UI RC3)

The front end presents a focused studio rather than a long sequence of independent cards:

- A compact introduction and clear **creation workspace** for selecting an image and configuring a real job.
- On completion the page scrolls to the **result workspace**, with tabs for **Artwork**, **Drawing**, **Comparison**, and **Technical data**.
- Artwork loads the real portrait and vector preview returned by the job; input-stage photos are available on demand.
- The Drawing tab lazily mounts the existing true SVG Path player. Playing, pausing, restarting, and changing speed do not modify the exported SVG or job data.
- Comparison uses the existing interactive image slider; Analysis reads the real Manifest and visualizes Greedy versus 2-opt pen-up travel.
- System diagnostics are available from a collapsed disclosure area instead of occupying the main presentation flow.

The header has no special “final demo” badge. Algorithm, API, metrics, and exported artifacts are unchanged from core frozen v1.0. The previous RC2 UI is available as the rollback baseline.

### RC4：完整画幅预览

作品视图默认以等比例、无遮挡方式显示 GAN 肖像和机器人 SVG 栅格预览。可切换「观察细节」（仅前端浏览器裁切），或点击进入完整大图；两种模式均不改变后端文件、矢量路径、指标或绘制顺序。

## RC5 前端精修（保留 RC4 结构）

- 首页流程卡片高度与文案优化，提升投影可读性；移除装饰性的课程演示标识。
- 完成任务后使用明确的「查看本次作品」按钮引导至结果，不改变任务轮询和后端 API。
- 结果大图支持上一阶段/下一阶段按钮及左右方向键切换，Esc 关闭；对话框内焦点可循环。
- 绘制视图提高未开始状态的参考轨迹对比度，播放时仍严格使用后端 SVG 顺序；播放控制与倍速按钮增加无障碍标签。
- 新增轻量状态反馈、键盘焦点样式、小屏和减少动态效果适配；不修改 PNG、SVG、Manifest 或任何算法参数。
