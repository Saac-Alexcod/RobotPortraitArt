param(
    [string]$BindHost = "127.0.0.1",
    [ValidateRange(1, 65535)][int]$Port = 8000,
    [string]$Distro = $env:ROBOT_PORTRAIT_APD_DISTRO,
    [string]$APDPython = $env:ROBOT_PORTRAIT_APD_PYTHON,
    [switch]$Reload,
    [switch]$RequireAPDrawingGAN,
    [switch]$SetupAPD,
    [switch]$SetupCartoon,
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"

# Setup always creates the managed environment, even if an older launcher left
# a Python override in this PowerShell session. Explicit parameters still win.
if ($SetupAPD -and -not $PSBoundParameters.ContainsKey("APDPython")) {
    $APDPython = ""
}
if ($SetupAPD -and ($CheckOnly -or $APDPython)) {
    throw "Use -SetupAPD without -CheckOnly or -APDPython."
}
if ($SetupCartoon -and $CheckOnly) { throw "Use -SetupCartoon without -CheckOnly." }

# Bootstrap only the project's Windows environment. WSL packages are opt-in.
if (-not (Test-Path $venvPython)) {
    if ($CheckOnly) { throw "Project .venv is missing. Run .\scripts\start_web.ps1 to create it." }
    Write-Host "[RobotPortraitArt] Creating Windows .venv..."
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -m venv (Join-Path $repoRoot ".venv")
    }
    elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv (Join-Path $repoRoot ".venv")
    }
    else { throw "Install Python 3.10+ for Windows, then run this script again." }
    if ($LASTEXITCODE -ne 0) { throw "Could not create Windows .venv. Python 3.10+ is required." }
}

$launcherArgs = @((Join-Path $PSScriptRoot "start_web.py"), "--host", $BindHost, "--port", "$Port")
if ($Distro) { $launcherArgs += @("--distro", $Distro) }
if ($APDPython) { $launcherArgs += @("--apd-python", $APDPython) }
if ($Reload) { $launcherArgs += "--reload" }
if ($RequireAPDrawingGAN) { $launcherArgs += "--require-apdrawinggan" }
if ($SetupAPD) { $launcherArgs += "--setup-apd" }
if ($SetupCartoon) { $launcherArgs += "--setup-cartoon" }
if ($CheckOnly) { $launcherArgs += "--check-only" }

Push-Location $repoRoot
try {
    & $venvPython @launcherArgs
    if ($LASTEXITCODE -ne 0) { throw "Web launcher exited with code $LASTEXITCODE. See the message above." }
}
finally { Pop-Location }
