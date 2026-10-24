param(
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8000,
    [string]$Distro = "Ubuntu-22.04",
    [string]$APDPython = "/home/winter/.venvs/robotportrait-apd/bin/python",
    [switch]$Reload
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    throw "Windows project virtualenv Python not found: $venvPython"
}

$wslExe = (Get-Command wsl.exe -ErrorAction Stop).Source

$env:ROBOT_PORTRAIT_APD_PYTHON = $APDPython
$env:PYTHONUTF8 = "1"

[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)

Write-Host ""
Write-Host "[RobotPortraitArt] Web backend startup"
Write-Host "  Repo:           $repoRoot"
Write-Host "  Windows Python: $venvPython"
Write-Host "  WSL distro:     $Distro"
Write-Host "  APD Python:     $APDPython"
Write-Host "  API:            http://${BindHost}:$Port"
Write-Host ""

$preflightCode = @'
import torch

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("PyTorch CUDA is unavailable.")

print("gpu:", torch.cuda.get_device_name(0))
'@

Write-Host "[RobotPortraitArt] Checking APDrawingGAN WSL/CUDA environment..."

# Feed Python code through stdin instead of `python -c` so PowerShell/wsl.exe
# quoting cannot strip the quotes inside the Python program.
$preflightCode | & $wslExe `
    -d $Distro `
    -- `
    $APDPython `
    -

if ($LASTEXITCODE -ne 0) {
    throw "APDrawingGAN WSL/CUDA preflight failed."
}

Write-Host ""
Write-Host "[RobotPortraitArt] APDrawingGAN WSL/CUDA preflight passed."
Write-Host "[RobotPortraitArt] Starting FastAPI..."
Write-Host ""

$uvicornArgs = @(
    "-m", "uvicorn",
    "robot_portrait.web.app:app",
    "--app-dir", (Join-Path $repoRoot "src"),
    "--host", $BindHost,
    "--port", "$Port"
)

if ($Reload) {
    $uvicornArgs += "--reload"
}

Push-Location $repoRoot
try {
    & $venvPython @uvicornArgs
}
finally {
    Pop-Location
}
