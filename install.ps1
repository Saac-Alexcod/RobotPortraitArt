<#
RobotPortraitArt first-run deployment assistant (Windows PowerShell 5.1+).
Run from project root:
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
No background installations; run again after any required reboot.
The final acceptance test performs real APDrawingGAN inference.
#>
[CmdletBinding()]
param(
    [string]$Distro = '',
    [string]$APDPython = '',
    [string]$Proxy = '',
    [switch]$Doctor,
    [switch]$CheckWSLOnly,
    [switch]$NoStart,
    [switch]$BaselineOnly
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$root = $PSScriptRoot
Set-Location -LiteralPath $root
$env:PYTHONUTF8 = '1'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$logDir = Join-Path $root 'outputs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$logFile = Join-Path $logDir 'first_run_setup.log'
$stepLogDir = Join-Path $logDir 'installer_logs'
New-Item -ItemType Directory -Path $stepLogDir -Force | Out-Null
$script:phase = 'preflight'
$script:systemPyForRunner = $null
$script:explicitDistro = [bool]$Distro
$script:selectedWslPython = ''
$transcriptStarted = $false
try {
    $proxyUri = $null
    $proxyHasCredentials = ($Proxy -and [Uri]::TryCreate($Proxy, [UriKind]::Absolute, [ref]$proxyUri) -and $proxyUri.UserInfo)
    if (-not $proxyHasCredentials) {
        Start-Transcript -LiteralPath (Join-Path $stepLogDir 'powershell_transcript.log') -Append -ErrorAction Stop | Out-Null
        $transcriptStarted = $true
    }
} 
catch { Write-Warning 'PowerShell transcript unavailable; detailed native command logs will still be saved.' }

function Step([string]$message) {
    if ($message -match '^[1-7]/7 ') { $script:phase = $message }
    Write-Host "`n[RobotPortraitArt] $message" -ForegroundColor Cyan
    Add-Content -LiteralPath $logFile -Value ((Get-Date -Format 's') + ' ' + $message) -Encoding UTF8
}
function Fail([string]$message) { throw ('Installation blocked: ' + $message + '. You may fix this and rerun install.ps1; existing installations are retained.') }
function RequireLastExit([string]$operation) {
    if ($LASTEXITCODE -ne 0) { Fail ($operation + ' failed with exit code ' + $LASTEXITCODE) }
}
function RunStage {
    param(
        [string]$Stage, [int]$Timeout, [string]$Exe,
        [string[]]$Arguments = @(), [string]$InputText = '',
        [switch]$AllowFailure
    )
    if (-not $script:systemPyForRunner) { Fail 'Diagnostic runner requires a discovered Windows Python' }
    $slug = ($Stage.ToLowerInvariant() -replace '[^a-z0-9]+', '_').Trim('_')
    $stageLog = Join-Path $stepLogDir ($slug + '.log')
    $runnerArgs = @((Join-Path $root 'scripts\installer_diagnostics.py'), '--step', $Stage,
              '--log', $stageLog, '--timeout', [string]$Timeout)
    $stdinFile = $null
    if ($InputText) {
        $stdinFile = Join-Path $stepLogDir ($slug + '.stdin.tmp')
        [System.IO.File]::WriteAllText($stdinFile, $InputText, (New-Object System.Text.UTF8Encoding($false)))
        $runnerArgs += @('--stdin-file', $stdinFile)
    }
    $runnerArgs += '--'
    $runnerArgs += $Exe
    $runnerArgs += $Arguments
    try {
        # Keep the process runner's status output out of this function's
        # success/failure return value (PowerShell sends uncaptured native
        # stdout into the pipeline, where it breaks Boolean conversion).
        & $script:systemPyForRunner @runnerArgs | Out-Host
        $code = $LASTEXITCODE
    } finally {
        if ($stdinFile -and (Test-Path -LiteralPath $stdinFile)) {
            Remove-Item -LiteralPath $stdinFile -Force -ErrorAction SilentlyContinue
        }
    }
    if ($code -eq 0) { return $true }
    if ($AllowFailure) { return $false }
    if ($code -eq 124) { Fail ('[RPA-E120] ' + $Stage + ' timed out after ' + $Timeout + 's; log: ' + $stageLog) }
    Fail ('[RPA-E121] ' + $Stage + ' exited with code ' + $code + '; log: ' + $stageLog)
}
function RefreshPath {
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
}
function FindWindowsPython {
    param([switch]$ForNewVenv)
    # Windows' backend and the WSL GAN use DIFFERENT Python installations.
    # Running project .venv with Python 3.14 may be perfectly valid even though
    # the pinned Linux torch 2.5.1 environment requires Python <= 3.12.
    # New Windows environments prefer the validated 3.10-3.12 range.
    $candidates = New-Object 'System.Collections.Generic.List[string]'
    $projectPy = Join-Path $root '.venv\Scripts\python.exe'
    if (-not $ForNewVenv -and (Test-Path -LiteralPath $projectPy)) {
        $candidates.Add($projectPy)
    }
    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        foreach ($version in @('3.11','3.12','3.10','3.14','3.13')) {
            try {
                # Avoid quoted string literals in -c: Windows PowerShell 5.1
                # may strip nested double quotes in native argv forwarding.
                $paths = @(& $launcher.Source ('-' + $version) -c 'import sys; print(sys.executable)' 2>$null)
                $launcherExit = $LASTEXITCODE
                if ($launcherExit -eq 0 -and $paths.Count) {
                    $candidatePath = ([string]($paths | Select-Object -Last 1)).Trim()
                    if ($candidatePath) { $candidates.Add($candidatePath) }
                }
            } catch { }
        }
        # Include vendor-tagged launcher installs such as
        # -V:Astral/CPython3.12.14 E:\DevTools\...\python.exe.
        try {
            $listing = @(& $launcher.Source '-0p' 2>$null)
            if ($LASTEXITCODE -eq 0) {
                foreach ($line in $listing) {
                    if ([string]$line -match '(?i)([a-z]:\\[^\r\n]*?python\.exe)') {
                        $candidates.Add($Matches[1].Trim())
                    }
                }
            }
        } catch { }
    }
    foreach ($version in @('Python311','Python312','Python310','Python314','Python313')) {
        if ($env:LOCALAPPDATA) {
            $candidates.Add((Join-Path $env:LOCALAPPDATA ('Programs\Python\' + $version + '\python.exe')))
        }
    }
    foreach ($name in @('python.exe','python3.exe')) {
        $found = Get-Command $name -All -ErrorAction SilentlyContinue
        foreach ($entry in @($found)) {
            if ($entry -and $entry.Source) { $candidates.Add($entry.Source) }
        }
    }
    foreach ($candidate in @($candidates | Select-Object -Unique)) {
        if (-not $candidate -or -not (Test-Path -LiteralPath $candidate -PathType Leaf)) { continue }
        try {
            if ($ForNewVenv) {
                $probeCode = 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)'
            } else {
                $probeCode = 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,14) else 1)'
            }
            & $candidate -c $probeCode 2>$null | Out-Null
            $probeExit = $LASTEXITCODE
            if ($probeExit -eq 0) {
                Write-Host ('[RPA] Reusing Windows Python: ' + $candidate)
                return $candidate
            }
        } catch {
            Write-Host ('[RPA] Skipping Python candidate: ' + $candidate)
        }
    }
    return $null
}
function InstallWinget([string]$id, [string]$name) {
    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (-not $winget) { Fail ($name + ' missing and winget not available; Windows App Installer must be installed') }
    Step ('Installing missing ' + $name + ' using Windows Package Manager')
    $safeName = ($name -replace '[^a-zA-Z0-9]+', '_')
    $stdout = Join-Path $stepLogDir ('winget_' + $safeName + '.out.log')
    $stderr = Join-Path $stepLogDir ('winget_' + $safeName + '.err.log')
    $arguments = 'install --id ' + $id + ' --exact --source winget --accept-package-agreements --accept-source-agreements --silent'
    $process = Start-Process -FilePath $winget.Source -ArgumentList $arguments -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $elapsed = 0
    while (-not $process.WaitForExit(15000)) {
        $elapsed += 15
        Write-Host ('[RPA] winget ' + $name + ' still running (' + $elapsed + 's). Logs: ' + $stdout)
        if ($elapsed -ge 1200) {
            & taskkill.exe /PID $process.Id /T /F 2>$null | Out-Null
            Fail ('[RPA-E120] winget ' + $name + ' exceeded 20 minutes. See ' + $stdout + ' and ' + $stderr)
        }
    }
    foreach ($logPath in @($stdout, $stderr)) {
        if (Test-Path $logPath) { Get-Content -LiteralPath $logPath -Tail 35 | ForEach-Object { Write-Host $_ } }
    }
    # PowerShell 5.1 can return a process object with an unpopulated
    # ExitCode property until it is refreshed; never print a blank exit=.
    $process.WaitForExit()
    $process.Refresh()
    $wingetExitCode = $process.ExitCode
    if ($null -eq $wingetExitCode) {
        Fail ('[RPA-E121] winget ' + $name + ' finished without an available exit code; check ' + $stdout + ' and ' + $stderr)
    }
    if ($wingetExitCode -ne 0) {
        Fail ('[RPA-E121] winget ' + $name + ' exited with ' + $wingetExitCode + '; check ' + $stdout + ' and ' + $stderr)
    }
    RefreshPath
}
function TestWSLDistroRegistered {
    # Registered and runnable are different states. A failed startup is NOT
    # evidence that an already-registered distro should be reinstalled.
    # WSL may emit UTF-16/embedded NUL characters when piped in PowerShell 5.1.
    try {
        $rawNames = @(& wsl.exe --list --quiet 2>$null)
        $exitCode = $LASTEXITCODE
    } catch {
        Fail ('[RPA-E241] Unable to list WSL distributions: ' + $_.Exception.Message +
              '. Run wsl --list --verbose in PowerShell; no installation attempted')
    }
    if ($exitCode -ne 0) {
        Fail ('[RPA-E241] wsl --list --quiet failed (exit ' + $exitCode +
              '). Check WSL service or Windows components; no installation attempted')
    }
    $names = @(
        foreach ($line in $rawNames) {
            $name = ([string]$line).Replace([string][char]0, '').Trim().Trim([char]0xFEFF)
            if ($name) { $name }
        }
    )
    return ($names -contains $Distro)
}
function ResolveWSLDistro {
    # Windows' wsl.exe list output varies between UTF-16/UTF-8 depending on
    # host and PowerShell. Python reads raw bytes, distinguishes broken WSL
    # from an empty distro list, and probes Python compatibility in each WSL2.
    $outFile = Join-Path $stepLogDir 'wsl_discovery.json'
    $script = Join-Path $root 'scripts\installer_compatibility.py'
    $args = @($script, '--json', $outFile)
    if ($script:explicitDistro) { $args += @('--prefer', $Distro) }
    $null = RunStage -Stage 'Discover compatible WSL distributions' -Timeout 600 -Exe $script:systemPyForRunner -Arguments $args
    $report = Get-Content -LiteralPath $outFile -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($entry in @($report.probes)) {
        if ($entry.name) {
            Write-Host ('[RPA] WSL candidate ' + $entry.name + ': Python ' +
                $entry.python_version + ', ready=' + $entry.runtime_ready +
                ', usable=' + $entry.usable)
        }
    }
    if ($report.selected -and $report.selected.name) {
        $script:Distro = [string]$report.selected.name
        if ($report.selected.python) { $script:selectedWslPython = [string]$report.selected.python }
        Write-Host ('[RPA] Reusing compatible WSL2 distribution: ' + $script:Distro) -ForegroundColor Green
    } elseif ($report.status -in @('not_installed','needs_fallback')) {
        if (-not $script:Distro) { $script:Distro = 'Ubuntu-22.04' }
        if ((@($report.registered) -contains $script:Distro)) {
            Fail ('[RPA-C110] Requested WSL distribution is installed but unusable. ' +
                  'Check ' + $outFile + '; do not reinstall an existing distribution')
        }
        Write-Host ('[RPA] No compatible installed WSL2 distro. ' + $script:Distro + ' will be installed.')
    } else {
        Fail ('[RPA-C110] No usable compatible WSL2 installation. Registered: ' +
             (@($report.registered) -join ', ') + '. Review ' + $outFile +
             '. A distro that cannot launch / has unsupported Python must not be silently reinstalled.')
    }
    return $report
}
function AssertWindowsDiskSpace {
    param([int]$MinGiB = 6)
    try {
        $drive = New-Object System.IO.DriveInfo ([System.IO.Path]::GetPathRoot($root))
        $available = [math]::Round($drive.AvailableFreeSpace / 1GB, 1)
        Write-Host ('[RPA] Project drive free: ' + $available + ' GiB')
        if ($drive.AvailableFreeSpace -lt ($MinGiB * 1GB)) {
            Fail ('[RPA-E280] Insufficient space on project drive (' + $available +
                  ' GiB free; ' + $MinGiB + ' GiB recommended). Free space or move the checkout.')
        }
    } catch {
        if ($_.Exception.Message -match 'RPA-E280') { throw }
        Write-Warning ('Unable to measure project drive free space: ' + $_.Exception.Message)
    }
}
function AssertWslDiskSpace {
    # Run only if the existing environment cannot already execute GAN.
    $diskScript = @'
avail=$(df -Pk "$HOME" | awk 'NR==2 {print $4}')
if [ -z "$avail" ]; then echo RPA_DISK_UNKNOWN; exit 3; fi
echo RPA_WSL_FREE_KIB=$avail
test "$avail" -ge 8388608
'@
    $spaceOk = RunStage -Stage 'Check WSL free disk space' -Timeout 45 -Exe 'wsl.exe' -Arguments @('-d',$Distro,'--','sh','-s') -InputText $diskScript -AllowFailure
    if (-not $spaceOk) {
        Fail ('[RPA-E281] WSL has less than 8 GiB available for a fresh PyTorch install ' +
              '(or disk check failed). Run wsl -d ' + $Distro +
              ' -- df -h ~; free space inside WSL, then rerun.')
    }
}

function ProbeWSL {
    # Only a shell exit status matters; diagnostic stdout is not a success flag.
    try {
        return [bool](RunStage -Stage 'Probe WSL readiness' -Timeout 60 -Exe 'wsl.exe' -Arguments @('-d', $Distro, '--', 'sh', '-c', 'true') -AllowFailure)
    } catch { return $false }
}
function ResolveNode {
    $node = Get-Command node.exe -ErrorAction SilentlyContinue
    if (-not $node) {
        $local = Join-Path $env:ProgramFiles 'nodejs\node.exe'
        if (Test-Path -LiteralPath $local) { return $local }
    }
    if ($node) { return $node.Source }
    return $null
}
function WslProxyArgs {
    if (-not $Proxy) { return @() }
    $uri = $null
    if (-not [Uri]::TryCreate($Proxy, [UriKind]::Absolute, [ref]$uri) -or $uri.Scheme -notin @('http', 'https')) {
        Fail 'Proxy must be a valid HTTP(S) URL'
    }
    # WSL NAT cannot access Windows localhost; try its host gateway instead.
    $wslAddress = $Proxy
    if ($uri.Host -in @('127.0.0.1', 'localhost')) {
        $networkMode = ''
        try { $networkMode = (& wsl.exe -d $Distro -- wslinfo --networking-mode 2>$null) -join ' ' } catch {}
        if ($networkMode.Trim() -eq 'mirrored') { return @('env', ('HTTPS_PROXY=' + $Proxy), ('HTTP_PROXY=' + $Proxy), ('https_proxy=' + $Proxy), ('http_proxy=' + $Proxy)) }
        $route = 'ip route show default 2>/dev/null' | & wsl.exe -d $Distro -- sh -s 2>$null
        if ($LASTEXITCODE -eq 0 -and ($route -join ' ') -match 'via\s+([0-9.]+)') {
            $hostIp = $Matches[1]
            $builder = New-Object -TypeName System.UriBuilder -ArgumentList $uri
            $builder.Host = $hostIp
            $wslAddress = $builder.Uri.AbsoluteUri.TrimEnd('/')
            Step 'WSL proxy uses discovered Windows gateway (not Windows localhost)'
        }
    }
    return @('env', ('HTTPS_PROXY=' + $wslAddress), ('HTTP_PROXY=' + $wslAddress), ('https_proxy=' + $wslAddress), ('http_proxy=' + $wslAddress))
}

try {
    if ($CheckWSLOnly) {
        Step 'Read-only WSL registration, version and Python compatibility check'
        $py = FindWindowsPython
        if (-not $py) { Fail 'No runnable Windows Python 3.10-3.14 could be located for read-only WSL discovery. Check py -0p and .venv\Scripts\python.exe; WSL was not modified.' }
        $script:systemPyForRunner = $py
        $discovery = ResolveWSLDistro
        if (-not $discovery.selected) {
            Fail ('No compatible WSL2 distribution was discovered; would select ' + $Distro + ' for installation')
        }
        Write-Host ('[RPA] PASS: ' + $Distro + ' existing compatible WSL2; no installation performed.') -ForegroundColor Green
        exit 0
    }
    if ($Doctor) {
        Step 'Read-only environment report'
        $py = FindWindowsPython
        if (-not $py) { Fail 'No suitable Windows Python found for diagnostics' }
        if (-not $Distro) { $Distro = 'Ubuntu-22.04' }; & $py (Join-Path $root 'scripts\doctor_environment.py') --distro $Distro
        exit $LASTEXITCODE
    }
    # Downloads/installer subprocesses inherit this process-local proxy only.
    if ($Proxy) {
        $env:HTTP_PROXY = $Proxy
        $env:HTTPS_PROXY = $Proxy
        $env:http_proxy = $Proxy
        $env:https_proxy = $Proxy
    }
    Step '1/7 Windows Python'
    AssertWindowsDiskSpace -MinGiB 5
    $systemPy = FindWindowsPython
    if (-not $systemPy) {
        InstallWinget 'Python.Python.3.11' 'Python 3.11'
        $systemPy = FindWindowsPython
        if (-not $systemPy) { Fail 'Python was installed but not discoverable; reopen PowerShell and rerun' }
    }
    $script:systemPyForRunner = $systemPy
    Step '2/7 Windows backend dependencies'
    $venv = Join-Path $root '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $venv) {
        try {
            & $venv -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,14) else 1)' 2>$null | Out-Null
            $venvCode = $LASTEXITCODE
        } catch { $venvCode = 1 }
        if ($venvCode -ne 0) {
            Fail '[RPA-C101] Existing .venv cannot run with a supported Windows Python (3.10-3.14). Back up .venv outside the project and rerun; do not uninstall global Python.'
        }
        Write-Host '[RPA] Existing project .venv is usable; checking installed backend dependencies.'
    } else {
        $creationPy = FindWindowsPython -ForNewVenv
        if (-not $creationPy) {
            InstallWinget 'Python.Python.3.11' 'Python 3.11'
            $creationPy = FindWindowsPython -ForNewVenv
        }
        if (-not $creationPy) {
            Fail '[RPA-E210] No compatible Python 3.10-3.12 for new .venv. Run py -0p and confirm a 64-bit Python 3.11 installation before retrying.'
        }
        $null = RunStage -Stage 'Create Windows virtual environment' -Timeout 240 -Exe $creationPy -Arguments @('-m', 'venv', (Join-Path $root '.venv'))
    }
    $requirements = Join-Path $root 'requirements-web.txt'
    $requirementsHash = (Get-FileHash -LiteralPath $requirements -Algorithm SHA256).Hash
    $pipStamp = Join-Path $root '.venv\.rpa-requirements-web.sha256'
    $depsOk = $false
    try {
        & $venv -c 'import fastapi,uvicorn,cv2,multipart,numpy,PIL,skimage,httpx2' 2>$null | Out-Null
        $importsOk = ($LASTEXITCODE -eq 0)
        if ($importsOk) {
            & $venv -m pip check 2>$null | Out-Null
            $depsOk = ($LASTEXITCODE -eq 0)
        }
    } catch { $depsOk = $false }
    if (-not $depsOk) {
        $null = RunStage -Stage 'Install Windows Python requirements' -Timeout 1800 -Exe $venv -Arguments @('-m', 'pip', 'install', '--disable-pip-version-check', '--retries', '4', '--timeout', '45', '-r', $requirements)
        Set-Content -LiteralPath $pipStamp -Value $requirementsHash -Encoding ASCII
    } else { Step 'Windows dependencies already ready; skip pip install' }
    $null = RunStage -Stage 'Validate Windows pip dependencies' -Timeout 120 -Exe $venv -Arguments @('-m', 'pip', 'check')

    Step '3/7 Node.js and Vue dependencies'
    # Do not upgrade or overwrite another project/user's global Node.js.
    # A compatible system Node is reused; otherwise download verified portable
    # Node into an ignored, project-local directory.
    $node = ResolveNode
    $nodeOk = $false
    if ($node) {
        try {
            $nodeVersion = & $node -p 'process.versions.node' 2>$null
            if ($LASTEXITCODE -eq 0) {
                $version = [Version]($nodeVersion | Select-Object -Last 1)
                $nodeOk = ($version.Major -gt 22 -or
                           ($version.Major -eq 22 -and $version.Minor -ge 12))
            }
        } catch { $nodeOk = $false }
    }
    if (-not $nodeOk) {
        Write-Host '[RPA-C103] Missing or incompatible Node; installing verified portable Node 22 LTS (global Node stays untouched)'
        $nodeArgs = @((Join-Path $root 'scripts\install_portable_node.py'))
        if ($Proxy) { $nodeArgs += @('--proxy', $Proxy) }
        $null = RunStage -Stage 'Install isolated compatible Node LTS' -Timeout 1500 -Exe $systemPy -Arguments $nodeArgs
        $node = Join-Path $root '.rpa_tools\node-v22.16.0\node.exe'
    }
    $npmPath = Join-Path (Split-Path -Parent $node) 'npm.cmd'
    if (-not (Test-Path -LiteralPath $npmPath)) {
        Fail '[RPA-E230] Compatible Node found, but matching npm.cmd is missing. Check local Node installation.'
    }
    $env:Path = (Split-Path -Parent $node) + ';' + $env:Path
    $frontend = Join-Path $root 'frontend'
    $lockHash = (Get-FileHash -LiteralPath (Join-Path $frontend 'package-lock.json') -Algorithm SHA256).Hash
    $npmStamp = Join-Path $frontend 'node_modules\.rpa-lock-sha256'
    if (-not (Test-Path -LiteralPath $npmStamp) -or ((Get-Content -LiteralPath $npmStamp -Raw).Trim() -ne $lockHash)) {
        $null = RunStage -Stage 'Install frontend npm dependencies' -Timeout 1800 -Exe $npmPath -Arguments @('--prefix', $frontend, 'ci', '--no-audit', '--no-fund')
        Set-Content -LiteralPath $npmStamp -Value $lockHash -Encoding ASCII
    } else { Step 'Frontend lockfile already installed; skip npm ci' }

    if ($BaselineOnly) {
        Step 'Baseline-only mode: omit WSL, CUDA and GAN model provisioning'
        if (-not $NoStart) { & (Join-Path $root 'scripts\start_all_baseline.ps1'); RequireLastExit 'Start baseline web services' }
        exit 0
    }

    Step '4/7 WSL2 and Ubuntu'
    if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
        Fail '[RPA-E240] wsl.exe unavailable. Enable Windows Subsystem for Linux and Virtual Machine Platform; reboot if prompted.'
    }
    $discovery = ResolveWSLDistro
    if ($discovery.selected) {
        Step ('WSL ' + $Distro + ' already registered and compatible; will not reinstall')
        if (-not (ProbeWSL)) {
            Fail ('[RPA-E242] WSL ' + $Distro + ' selected but cannot start. ' +
                  'Inspect probe_wsl_readiness.log; do not reinstall an existing distro.')
        }
    } elseif (TestWSLDistroRegistered) {
        Fail ('[RPA-C110] ' + $Distro + ' is registered but incompatible; no reinstall permitted')
    } else {
        try {
            $systemDrive = New-Object System.IO.DriveInfo ($env:SystemDrive + '\')
            if ($systemDrive.AvailableFreeSpace -lt (10 * 1GB)) {
                Fail '[RPA-E282] System drive has less than 10 GiB for a new WSL distro. Free C: space first.'
            }
        } catch { if ($_.Exception.Message -match 'RPA-E282') { throw } }
        Step ('Installing verified fallback WSL distribution ' + $Distro +
              ' (admin approval/reboot may be necessary)')
        $process = Start-Process -FilePath 'wsl.exe' -Verb RunAs -ArgumentList (
           '--install --distribution ' + $Distro + ' --no-launch') -PassThru
        $elapsed = 0
        while (-not $process.WaitForExit(15000)) {
            $elapsed += 15
            Write-Host ('[RPA] WSL install still running (' + $elapsed + 's); UAC/reboot may be required.')
            if ($elapsed -ge 1200) { Fail '[RPA-E120] WSL install exceeded 20 minutes. Check admin/UAC and Windows reboot.' }
        }
        if ($process.ExitCode -ne 0) { Fail ('[RPA-E243] WSL install exited ' + $process.ExitCode) }
        if (-not (TestWSLDistroRegistered) -or -not (ProbeWSL)) {
            Fail ('[RPA-E243] WSL installed but not ready. Reboot/initialize Ubuntu user, then rerun install.ps1')
        }
    }

    Step '5/7 WSL Python / CUDA-enabled PyTorch'
    # Reuse the existing Python if it already has the project GAN runtime.
    $selectedPy = $APDPython
    if (-not $selectedPy -and $script:selectedWslPython) { $selectedPy = $script:selectedWslPython }
    if (-not $selectedPy) {
        $findScript = @'
p="$HOME/.venvs/robotportrait-apd/bin/python"
if [ -x "$p" ]; then printf "%s" "$p"; else command -v python3 || true; fi
'@
        $null = RunStage -Stage 'Discover WSL Python' -Timeout 60 -Exe 'wsl.exe' -Arguments @('-d', $Distro, '--', 'sh', '-s') -InputText $findScript
        $existing = Get-Content -LiteralPath (Join-Path $stepLogDir 'discover_wsl_python.log') | Where-Object { $_ -match '^/' }
        $selectedPy = [string]($existing | Select-Object -Last 1)
        $selectedPy = $selectedPy.Trim()
    }
    # Single inference-specific probe shared by reuse and final runtime validation.
    # Visdom is training-only; lacking it must not invalidate a working test.py runtime.
    $runtimeProbe = (Get-Content -LiteralPath (Join-Path $root 'scripts\apd_runtime_probe.py') -Raw -Encoding UTF8).Replace("`r`n", "`n")
    $ready = $false
    if ($selectedPy) {
        try {
            $ready = [bool](RunStage -Stage 'Probe WSL PyTorch CUDA' -Timeout 90 -Exe 'wsl.exe' -Arguments @('-d', $Distro, '--', $selectedPy, '-') -InputText $runtimeProbe -AllowFailure)
        } catch { $ready = $false }
    }
    if (-not $ready) {
        AssertWslDiskSpace
        if ($APDPython) { Fail ('[RPA-E250] Explicit WSL Python failed inference runtime probe; see ' + (Join-Path $stepLogDir 'probe_wsl_pytorch_cuda.log') + '. No changes were made to this interpreter. Omit -APDPython to allow installing a separate project venv, or repair the indicated required dependency.') }
        # apt operates only within the selected Ubuntu distro.
        $aptScript = @'
set -eu
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y python3 python3-venv python3-pip ca-certificates
for p in python3.11 python3.10 python3.12 python3.9; do
  if command -v "$p" >/dev/null 2>&1; then
    pkg="$p-venv"
    if apt-cache show "$pkg" >/dev/null 2>&1; then apt-get install -y "$pkg"; fi
  fi
done
'@
        $proxyArgs = WslProxyArgs
        $aptCmd = @('-d', $Distro, '-u', 'root', '--') + $proxyArgs + @('sh', '-s')
        $null = RunStage -Stage 'Install WSL apt Python prerequisites' -Timeout 1800 -Exe 'wsl.exe' -Arguments $aptCmd -InputText $aptScript
        $wslScript = (Get-Content -LiteralPath (Join-Path $root 'scripts\install_wsl_runtime.sh') -Raw -Encoding UTF8).Replace("`r`n", "`n")
        $proxyArgs = WslProxyArgs
        $cmd = @('-d', $Distro, '--') + $proxyArgs + @('sh', '-s')
        $null = RunStage -Stage 'Install WSL CUDA PyTorch packages' -Timeout 5400 -Exe 'wsl.exe' -Arguments $cmd -InputText $wslScript
        $installedLog = Join-Path $stepLogDir 'install_wsl_cuda_pytorch_packages.log'
        $probe = @(Get-Content -LiteralPath $installedLog | Where-Object { $_ -match '^RPA_PYTHON=/' })
        if (-not $probe.Count) {
            Fail ('[RPA-C102] WSL installer did not output its selected Python path. See ' + $installedLog)
        }
        $selectedPy = (([string]($probe | Select-Object -Last 1)) -replace '^RPA_PYTHON=', '').Trim()
    }
    if (-not $selectedPy) { Fail 'No functional WSL Python found' }
    # Re-run the same inference probe, rather than a stricter training-dependency check.
    $null = RunStage -Stage 'Validate APDrawingGAN inference runtime' -Timeout 120 -Exe 'wsl.exe' -Arguments @('-d',$Distro,'--',$selectedPy,'-') -InputText $runtimeProbe

    Step '6/7 Download and validate models'
    $null = RunStage -Stage 'Restore and validate bundled APDrawingGAN source' -Timeout 180 -Exe $venv -Arguments @((Join-Path $root 'scripts\ensure_apdrawinggan_source.py'))
    $null = RunStage -Stage 'Back up corrupt ONNX assets' -Timeout 300 -Exe $venv -Arguments @((Join-Path $root 'scripts\repair_model_files.py'))
    $args = @((Join-Path $root 'scripts\bootstrap_models.py'), '--download', '--download-apd')
    if ($Proxy) { $args += @('--proxy', $Proxy) }
    $null = RunStage -Stage 'Download verify official APDrawingGAN model files' -Timeout 7200 -Exe $venv -Arguments $args
    $null = RunStage -Stage 'Download verify GANs N Roses model' -Timeout 7200 -Exe $venv -Arguments @((Join-Path $root 'scripts\setup_cartoon.py'))
    $null = RunStage -Stage 'APDrawingGAN final environment preflight' -Timeout 180 -Exe $venv -Arguments @((Join-Path $root 'scripts\check_apdrawinggan.py'), '--distro', $Distro, '--python', $selectedPy)

    Step '7/7 Real APDrawingGAN and GANs N Roses image generation acceptance tests'
    $env:ROBOT_PORTRAIT_APD_DISTRO = $Distro
    $env:ROBOT_PORTRAIT_APD_PYTHON = $selectedPy
    $smoke = Join-Path $root 'scripts\verify_first_run.py'
    $null = RunStage -Stage 'Real APDrawingGAN image generation' -Timeout 1800 -Exe $venv -Arguments @($smoke, '--distro', $Distro, '--python', $selectedPy)
    $null = RunStage -Stage 'Real GANs N Roses image generation' -Timeout 1800 -Exe $venv -Arguments @($smoke, '--distro', $Distro, '--python', $selectedPy, '--method', 'cartoon')

    if (-not $NoStart) {
        Step 'Starting both services'
        $null = RunStage -Stage 'Start backend and frontend' -Timeout 240 -Exe 'powershell.exe' -Arguments @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $root 'scripts\start_all.ps1'), '-Distro', $Distro, '-APDPython', $selectedPy)
    }
    Step 'PASS: Both GAN methods generated real results and services are configured'
    Write-Host '[PASS] First-run acceptance completed.' -ForegroundColor Green
} catch {
    Write-Host ''
    $reason = $_.Exception.Message
    Write-Host ('[FAILED] ' + $reason) -ForegroundColor Red
    Write-Host ('Stage: ' + $script:phase) -ForegroundColor Yellow
    Write-Host ('Summary log: ' + $logFile) -ForegroundColor Yellow
    Write-Host ('Detailed step logs: ' + $stepLogDir) -ForegroundColor Yellow
    Write-Host ('Help: ' + (Join-Path $root 'docs\INSTALL_TROUBLESHOOTING.md')) -ForegroundColor Yellow
    if ($reason -match 'RPA-C10[1-9]|RPA-C110') { Write-Host 'RPA-COMPAT: Interpreter/distro mismatch. Existing installations are preserved. See wsl_discovery.json and the compatibility guide.' }
    elseif ($reason -match 'RPA-E28[01]|No space left|WinError 112') { Write-Host 'RPA-E280: Disk capacity issue. Check project drive and WSL df -h ~; clear at least 8 GiB before retry.' }
    elseif ($reason -match 'timed out|RPA-E120') { Write-Host 'RPA-E120: Step timed out. Check connection, proxy, or WSL. Inspect detailed log; rerun to resume.' }
    elseif ($script:phase -match '1/7|2/7') { Write-Host 'RPA-E210: Windows Python/pip issue. Check Python installation, venv, network proxy and requirements-web.txt.' }
    elseif ($script:phase -match '3/7') { Write-Host 'RPA-E230: Node/npm issue. Check npm registry, proxy, disk space and package-lock.json.' }
    elseif ($script:phase -match '4/7') { Write-Host 'RPA-E240: WSL issue. Check virtualization, UAC, Windows restart and Ubuntu first launch.' }
    elseif ($script:phase -match '5/7') { Write-Host 'RPA-E250: PyTorch/CUDA issue. Check WSL GPU driver and PyTorch package downloads.' }
    elseif ($script:phase -match '6/7') { Write-Host 'RPA-E260: Model issue. Check HTTPS, proxy, integrity and available disk space.' }
    elseif ($script:phase -match '7/7') { Write-Host 'RPA-E270: Inference or web startup issue. See outputs/first_run_validation/*.log and start_backend_and_frontend.log.' }
    else { Write-Host 'Use the detailed step log to identify the cause; rerun after fixing it.' }
    Write-Host 'Recovery: rerun the SAME install.ps1 command; verified components are reused.'
    exit 1
} finally {
    if ($transcriptStarted) { try { Stop-Transcript | Out-Null } catch { } }
}
