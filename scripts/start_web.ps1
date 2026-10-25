param(
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8000,
    [string]$Distro = "Ubuntu-22.04",
    [string]$APDPython = "",
    [switch]$Reload,
    [switch]$RequireAPDrawingGAN
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    throw "Windows project virtualenv Python not found: $venvPython"
}

# Automatically prefer the user's existing dedicated WSL virtualenv.
# Do not assume another contributor has /home/winter on their computer.
if ([string]::IsNullOrWhiteSpace($APDPython)) {
    $APDPython = 'python3'
    try {
        $findPython = @'
p="$HOME/.venvs/robotportrait-apd/bin/python"
if [ -x "$p" ]; then printf "%s" "$p"; else command -v python3; fi
'@
        $detected = $findPython | & wsl.exe -d $Distro -- sh -s 2>$null
        if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($detected)) {
            $APDPython = ($detected | Select-Object -Last 1).Trim()
        }
    } catch {
        # Baseline startup must not depend on a working WSL environment.
    }
}
$env:ROBOT_PORTRAIT_APD_DISTRO = $Distro
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

# This is a read-only preflight. Never install packages or download models on startup.
$requiredModels = @(
    (Join-Path $repoRoot 'third_party\models\face_detection_yunet_2023mar.onnx'),
    (Join-Path $repoRoot 'third_party\models\human_segmentation_pphumanseg_2023mar.onnx')
)
foreach ($model in $requiredModels) {
    if (-not (Test-Path $model)) {
        Write-Warning "Optional model missing: $model"
        Write-Warning 'Run scripts/setup_environment.ps1 -DownloadModels to set up official ONNX files.'
    }
}
Write-Host "[RobotPortraitArt] Checking APDrawingGAN WSL/CUDA environment..."

$preflightPassed = $false
$preflightFailure = $null

try {
    $wslExe = (Get-Command wsl.exe -ErrorAction Stop).Source

    # Feed Python code through stdin instead of `python -c` so PowerShell/wsl.exe
    # quoting cannot strip the quotes inside the Python program.
    $preflightCode | & $wslExe `
        -d $Distro `
        -- `
        $APDPython `
        -

    if ($LASTEXITCODE -eq 0) {
        $preflightPassed = $true
    }
    else {
        $preflightFailure = (
            "APDrawingGAN WSL/CUDA preflight exited with code " +
            "$LASTEXITCODE."
        )
    }
}
catch {
    $preflightFailure = $_.Exception.Message
}

if ($preflightPassed) {
    Write-Host ""
    Write-Host "[RobotPortraitArt] APDrawingGAN WSL/CUDA preflight passed."
}
else {
    if ([string]::IsNullOrWhiteSpace($preflightFailure)) {
        $preflightFailure = "unknown error"
    }

    $message = (
        "APDrawingGAN WSL/CUDA preflight failed: " +
        $preflightFailure
    )

    if ($RequireAPDrawingGAN) {
        throw $message
    }

    Write-Warning $message
    Write-Warning (
        "FastAPI will continue in degraded mode. " +
        "Baseline remains available; /api/methods reports APDrawingGAN status."
    )
}

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
