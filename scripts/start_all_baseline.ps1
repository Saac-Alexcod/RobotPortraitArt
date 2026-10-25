<# Fallback: baseline-only startup without installing WSL or models. #>
[CmdletBinding()]
param([int]$ApiPort = 8000, [int]$WebPort = 5173)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$portableNode = Join-Path $root '.rpa_tools\node-v22.16.0'
if (Test-Path -LiteralPath (Join-Path $portableNode 'node.exe')) {
    $env:Path = $portableNode + ';' + $env:Path
}
$backend = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $backend)) { throw 'Missing Windows Python environment' }
$apiCmd = '/k "' + $backend + '" -m uvicorn robot_portrait.web.app:app --app-dir src --host 127.0.0.1 --port ' + $ApiPort
$webCmd = '/k cd /d "' + (Join-Path $root 'frontend') + '" && npm.cmd run dev -- --host 127.0.0.1 --port ' + $WebPort
Start-Process -FilePath cmd.exe -ArgumentList $apiCmd -WorkingDirectory $root
Start-Process -FilePath cmd.exe -ArgumentList $webCmd -WorkingDirectory (Join-Path $root 'frontend')
Write-Host "Baseline-only startup launched: http://127.0.0.1:$WebPort"
