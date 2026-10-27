param(
    [ValidateSet('all', 'apdrawinggan', 'cartoon')][string]$Model = 'all',
    [string]$InputImage,
    [string]$OutputDir,
    [string]$Distro = $env:ROBOT_PORTRAIT_APD_DISTRO,
    [string]$APDPython,
    [int]$Seed = 42,
    [switch]$Setup
)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot '.venv\Scripts\python.exe'
if ($Setup -and $APDPython) { throw 'Use -Setup without -APDPython.' }
if (-not (Test-Path -LiteralPath $venvPython)) {
    if (-not $Setup) { throw 'Missing .venv. Run this script with -Setup first.' }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -m venv (Join-Path $repoRoot '.venv')
    } else {
        & python -m venv (Join-Path $repoRoot '.venv')
    }
    if ($LASTEXITCODE -ne 0) { throw 'Could not create Python environment.' }
}
$runArgs = @((Join-Path $PSScriptRoot 'run_models.py'), '--model', $Model, '--seed', "$Seed")
if ($InputImage) { $runArgs += @('--input', $InputImage) }
if ($OutputDir) { $runArgs += @('--output-dir', $OutputDir) }
if ($Distro) { $runArgs += @('--distro', $Distro) }
if ($APDPython) { $runArgs += @('--apd-python', $APDPython) }
if ($Setup) { $runArgs += '--setup' }
& $venvPython @runArgs
if ($LASTEXITCODE -ne 0) { throw "Model runner failed with exit code $LASTEXITCODE." }
