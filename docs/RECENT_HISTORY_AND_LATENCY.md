# Local history and request timing (v1.0.1 feature note)

This document describes the features integrated into the public `v1.0.1` source tag. It does not modify or measure the historical 2026-10-19 face benchmark, and neither history recovery nor preflight caching accelerates GPU inference itself. The browser interaction was reviewed during development; quantitative end-to-end speedups have **not** been established by a controlled timing study.

## Recent inputs and tasks

- Uploaded or selected images are stored **locally in browser IndexedDB** (at most six; oldest entries are removed). A recent entry contains the image bytes, its original filename, and a reference to its latest created job.
- Click a recent entry to restore that image. If the associated job is still present under the backend `outputs/web_jobs/` directory, its metadata and output viewer are reloaded via existing `GET /api/jobs/{id}` and Manifest APIs. If the backend no longer has the job, the image can still be restored.
- Users can remove individual entries or clear the browser history. Neither action deletes backend jobs or model results. Browser storage may be unavailable (e.g. incognito/privacy settings), in which case entries last only until the page is closed.
- No new browser-to-server history endpoint is introduced. Previously completed tasks made before this feature are **not** automatically added to local history.
- Portrait images can contain sensitive personal data: local browser history should be used on trusted/private computers, and cleared after sharing a machine.

## Latency model

- The frontend records the elapsed time between submitting `POST /api/jobs` and receiving its response. This includes upload transfer, API validation, and preflight when necessary. It is not equivalent to the pipeline's inference time or complete wall-clock user wait.
- The backend temporarily reuses its default APDrawingGAN preflight result across method discovery and task creation. Healthy status TTL: 30 seconds. Unavailable status TTL: 5 seconds. `GET /api/methods?refresh=true` forces a live recheck. When no cached result is present, preflight runs on a thread pool so the FastAPI event loop stays responsive.
- The GPU worker remains **single-threaded**. Pipeline execution performs its own safety checks, and repeated model startup or heavyweight inference remains outside the status cache. This design optimizes repeated submissions but does not claim to speed up cold inference.

## Verification and limits

- Python tests: web API default status is called once across repeated discovery and task submission, but called again on explicit refresh.
- Client-side IndexedDB state is implementation-checked and TS syntax/type checked. At pre-release time full Vue build and real-browser automation were not completed in the isolated test environment. The user subsequently confirmed the Windows functionality before publishing v1.0.1; no independent browser automation or controlled latency benchmark is archived here.
