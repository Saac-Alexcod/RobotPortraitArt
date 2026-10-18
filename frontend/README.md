# RobotPortraitArt Web Demo

V1 establishes the Vue 3 + TypeScript frontend skeleton.

## Requirements

- Node.js `^20.19.0` or `>=22.12.0`
- npm 10+

## Install

```powershell
cd frontend
npm install
```

## Validate

```powershell
npm run typecheck
npm run build
```

## Develop

Start the FastAPI backend from the repository root:

```powershell
$env:ROBOT_PORTRAIT_APD_PYTHON = "/home/winter/.venvs/robotportrait-apd/bin/python"

python -m uvicorn `
  robot_portrait.web.app:app `
  --app-dir src `
  --host 127.0.0.1 `
  --port 8000
```

Then start Vite in another terminal:

```powershell
cd frontend
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

Vite proxies `/api/*` to `http://127.0.0.1:8000`, so later frontend API calls can use relative URLs without adding CORS configuration for local development.

## V1 scope

- Vue 3 + TypeScript + Vite scaffold
- responsive Web Demo shell
- reusable stage card component
- `/api` development proxy
- no backend requests yet

V2 will add a typed API client for `/api/health` and `/api/methods`.
