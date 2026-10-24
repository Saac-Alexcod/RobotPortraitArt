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
