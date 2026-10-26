<#
RobotPortraitArt opt-in bootstrap. Default is read-only diagnostics.
No automatic system changes, proxy changes, or dependency installation.
#>
[CmdletBinding()]
param(
    [string]$Distro = 'Ubuntu-22.04',
    [string]$APDPython = '',
    [switch]$InstallWindowsDeps,
    [switch]$DownloadModels,
    [switch]$DownloadAPD,
    [string]$APDChecksum = '',
    [string]$Proxy = '',
    [switch]$NoProxy
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $root '.venv\Scripts\python.exe'
$env:PYTHONUTF8 = '1'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
if ($Proxy -and $NoProxy) { throw 'Use either -Proxy or -NoProxy, not both.' }

# Prefer the project venv, otherwise locate an existing system interpreter.
$python = $null
if (Test-Path -LiteralPath $venv) {
    $python = $venv
} else {
    $pyLauncher = Get-Command 'py.exe' -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        $candidate = & $pyLauncher.Source -3.11 -c 'import sys;print(sys.executable)' 2>$null
        if ($LASTEXITCODE -eq 0 -and $candidate) { $python = $candidate[-1].Trim() }
    }
    if (-not $python) {
        $systemPython = Get-Command 'python.exe' -ErrorAction SilentlyContinue
        if ($systemPython) {
            $probe = & $systemPython.Source -c 'import sys;print(sys.executable)' 2>$null
            if ($LASTEXITCODE -eq 0 -and $probe) { $python = $probe[-1].Trim() }
        }
    }
}
if (-not $python) {
    Write-Host '[MISSING] Windows Python. Install Python 3.11 then rerun setup.' -ForegroundColor Yellow
    Write-Host 'https://www.python.org/downloads/'
    exit 2
}
Write-Host "[Python] $python"

if ($InstallWindowsDeps) {
    if (-not (Test-Path -LiteralPath $venv)) {
        Write-Host '[Install] Creating Windows venv...'
        & $python -m venv (Join-Path $root '.venv')
        if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
    }
    Write-Host '[Install] Installing Windows backend requirements...'
    $pipArgs = @('-m', 'pip', 'install', '--disable-pip-version-check', '-r', (Join-Path $root 'requirements-web.txt'))
    if ($Proxy) { $pipArgs += @('--proxy', $Proxy) }
    & $venv @pipArgs
    if ($LASTEXITCODE -ne 0) { throw 'Windows dependency installation failed. Check network/proxy/pip index.' }
    $python = $venv
}

$downloadArgs = @()
if ($DownloadModels) { $downloadArgs += '--download' }
if ($DownloadAPD) { $downloadArgs += '--download-apd' }
if ($Proxy) { $downloadArgs += @('--proxy', $Proxy) }
if ($NoProxy) { $downloadArgs += '--no-proxy' }
if ($APDChecksum) { $downloadArgs += @('--apd-sha256', $APDChecksum) }
if ($downloadArgs.Count -gt 0) {
    Write-Host '[Models] Checking and downloading requested model weights...'
    & $python (Join-Path $PSScriptRoot 'bootstrap_models.py') @downloadArgs
    if ($LASTEXITCODE -ne 0) { throw 'Model download/validation failed. Existing validated models remain unchanged.' }
}

$doctorArgs = @('--distro', $Distro)
if ($APDPython) { $doctorArgs += @('--python', $APDPython) }
Write-Host ''
Write-Host '[Doctor] Checking Windows, WSL, Python, PyTorch/CUDA, and models...'
& $python (Join-Path $PSScriptRoot 'doctor_environment.py') @doctorArgs
if ($LASTEXITCODE -ne 0) { throw 'Diagnostic script failed' }
Write-Host ''
Write-Host 'If WSL2 is missing, see: https://learn.microsoft.com/windows/wsl/install'
Write-Host 'If CUDA torch is missing, use the official PyTorch selector inside the WSL venv.'
Write-Host 'Nothing was installed in WSL and no global proxy settings were changed.'
Write-Host 'For WSL network troubleshooting: https://learn.microsoft.com/windows/wsl/networking'
