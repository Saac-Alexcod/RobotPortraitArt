# RobotPortraitArt frontend

Vue 3 + TypeScript + Vite interface for the RobotPortraitArt FastAPI service.

## Run locally

Backend first, from the repository root:

```powershell
.\scripts\start_web.ps1
```

Frontend in a separate terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Open `http://127.0.0.1:5173` (or the address printed by Vite). The backend launcher selects an available port starting at 8000 and writes `.cache/web-runtime.json`. Vite reads that file before each API request, so backend port changes require no edits or frontend restart. If several backends run, the most recently started instance is used. To connect to a manually launched backend, set `ROBOT_PORTRAIT_API_URL` before starting Vite; this overrides automatic discovery.

The upload panel now includes a **four-image sample gallery**. Clicking a sample selects a real JPEG `File` for the existing upload flow; manual file selection and drag-and-drop still work. Sample files are mirrored in `frontend/public/examples/` for Vite static serving and originate from [`../data/examples/`](../data/examples/). Baseline can run without APDrawingGAN model weights. Advanced GAN mode is only enabled when reported available by the backend.

## Production checks

```powershell
npm run typecheck
npm run build
```

## UI capabilities

- Upload via file selection, drag-and-drop, or one-click selection of four bundled synthetic portraits, with local image preview.
- Select APDrawingGAN/Baseline and Centerline/Contour according to API availability.
- Submit generation jobs and track queued/running/completed/failed states.
- Inspect generated Portrait / Robot Vector, preprocessed and aligned inputs.
- Compare processing stages with an interactive A/B slider.
- Play the true server-generated SVG path order with pen-down and pen-up visualization.
- Inspect Manifest metrics and download PNG, SVG and JSON results.

The browser does not reimplement the GAN, vectorization, or path-planning algorithms. Robot drawing is a visualization of the generated SVG trajectory, **not a physical robot hardware controller**.

See the [project README](../README.md) for backend setup, model requirements, API details and limitations.

### Sample gallery and full-frame image preview

- The four synthetic portrait examples are bundled under `frontend/public/examples/`, byte-for-byte identical to the originals in `data/examples/`.
- Clicking a sample selects the same JPEG `File` object used by the manual upload path. The upload-stage preview uses `object-fit: contain` and displays the entire original image rather than clipping a square portrait to the wide card. Use **View full image** (shown as “查看整图” in the UI) to open a larger preview; press `Esc` or the close button to dismiss it.
- The preview does not modify, resize, or crop the image submitted to FastAPI. The profile example is a challenge input; successful APDrawingGAN preprocessing and GPU generation are not guaranteed for every example.
- When replacing the examples, keep the copies in the CLI and Vite public directories in sync. The corresponding consistency tests detect stale or missing assets.

### Local history and submission latency

- The workspace keeps up to **six recent input images** in this browser's IndexedDB, together with the latest related job ID. Select an entry under **最近使用** to restore the image and (if the backend still has it under `outputs/web_jobs/`) reload its job/results. This does **not** upload photos to an additional service or save results in the browser. Delete entries individually or clear the local list. On private browsing or blocked browser storage, history may survive only until the page is closed. Existing jobs created before enabling this feature are not backfilled automatically.
- After submitting a task, the UI reports **server acceptance latency** separately from real inference/vectorization time. The backend caches default WSL/CUDA preflight status for up to 30 seconds (5 seconds on an unavailable result) so `GET /api/methods` and `POST /api/jobs` avoid repeat probes. "重新检查" forces an immediate refresh. The worker still validates availability when actually running the pipeline. A job already queued behind another job may still wait; this change does not change the single-GPU worker queue or speed up inference itself.
