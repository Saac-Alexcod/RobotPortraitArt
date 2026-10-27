<# Start both web services after environment setup; never installs software. #>
[CmdletBinding()]
param(
    [string]$Distro = 'Ubuntu-22.04',
    [string]$APDPython = '',
    [int]$ApiPort = 8000,
    [int]$WebPort = 5173
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$portableNode = Join-Path $root '.rpa_tools\node-v22.16.0'
if (Test-Path -LiteralPath (Join-Path $portableNode 'node.exe')) {
    $env:Path = $portableNode + ';' + $env:Path
}
$py = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $py)) { throw 'Windows .venv missing; run install.ps1' }
$npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
if (-not $npm) { throw 'npm.cmd not found; run install.ps1' }

function Test-Port([int]$Port) {
    try { $tcp = New-Object Net.Sockets.TcpClient('127.0.0.1', $Port); $tcp.Dispose(); return $true }
    catch { return $false }
}
if (-not (Test-Port $ApiPort)) {
    $script = Join-Path $root 'scripts\start_web.ps1'
    $cmd = '-NoProfile -ExecutionPolicy Bypass -File "' + $script + '" -Distro "' + $Distro + '" -APDPython "' + $APDPython + '" -Port ' + $ApiPort + ' -RequireAPDrawingGAN'
    Start-Process -FilePath 'powershell.exe' -WorkingDirectory $root -ArgumentList $cmd
} else { Write-Host '[API] Existing listener on port' $ApiPort }
if (-not (Test-Port $WebPort)) {
    $cmd = '/k cd /d "' + (Join-Path $root 'frontend') + '" && npm.cmd run dev -- --host 127.0.0.1 --port ' + $WebPort
    Start-Process -FilePath 'cmd.exe' -WorkingDirectory (Join-Path $root 'frontend') -ArgumentList $cmd
} else { Write-Host '[Web] Existing listener on port' $WebPort }
$apiReady = $false
for ($attempt = 0; $attempt -lt 40; $attempt++) {
    try {
        $health = Invoke-RestMethod -Uri ('http://127.0.0.1:' + $ApiPort + '/api/health') -TimeoutSec 2
        if ($health.service -eq 'robot-portrait-api' -and $health.status -eq 'ok') { $apiReady = $true; break }
    } catch {}
    Start-Sleep -Seconds 1
}
if (-not $apiReady) { throw 'API did not become healthy; inspect backend PowerShell window' }
$methods = Invoke-RestMethod -Uri ('http://127.0.0.1:' + $ApiPort + '/api/methods?refresh=true') -TimeoutSec 60
$gan = @($methods.methods | Where-Object { $_.name -eq 'apdrawinggan' -and $_.available })
if ($gan.Count -ne 1) { throw 'Running FastAPI did not report APDrawingGAN available' }
$cartoon = @($methods.methods | Where-Object { $_.name -eq 'cartoon' -and $_.available })
if ($cartoon.Count -ne 1) { throw 'Running FastAPI did not report GANs N Roses available' }
$webReady = $false
for ($attempt = 0; $attempt -lt 40; $attempt++) {
    try {
        $page = Invoke-WebRequest -Uri ('http://127.0.0.1:' + $WebPort) -UseBasicParsing -TimeoutSec 2
        if ($page.StatusCode -eq 200 -and $page.Content -match 'id=.app.') { $webReady = $true; break }
    } catch {}
    Start-Sleep -Seconds 1
}
if (-not $webReady) { throw 'Frontend Vite did not become healthy; inspect frontend command window' }
Write-Host ('API: http://127.0.0.1:' + $ApiPort)
Write-Host ('Web: http://127.0.0.1:' + $WebPort)
Write-Host '[PASS] API health endpoint responded. Browser should open Web URL.' -ForegroundColor Green
