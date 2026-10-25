"""Static Windows installer regression checks, not a Windows execution test.

The installer itself runs only on Windows PowerShell; validate the policy and
entry points here, and complete the real test using -CheckWSLOnly and -NoStart.
"""
from pathlib import Path

SCRIPT = (Path(__file__).parents[1] / 'install.ps1').read_text(encoding='utf-8')


def test_python_candidates_include_launcher_and_vendor_paths():
    assert "('3.11','3.12','3.10','3.14','3.13')" in SCRIPT
    assert "'-0p'" in SCRIPT
    assert "Programs\\Python\\" in SCRIPT
    assert "Get-Command $name -All" in SCRIPT


def test_windows_venv_is_not_rejected_just_because_it_is_python_314():
    assert "(3,10) <= sys.version_info[:2] <= (3,14)" in SCRIPT
    assert "Existing project .venv is usable" in SCRIPT
    assert 'import fastapi,uvicorn,cv2,multipart,numpy,PIL,skimage,httpx2' in SCRIPT
    assert '-m pip check' in SCRIPT


def test_new_venv_prefer_supported_python_and_existing_venv_not_recreated():
    assert 'FindWindowsPython -ForNewVenv' in SCRIPT
    assert "(3,10) <= sys.version_info[:2] <= (3,12)" in SCRIPT
    assert 'if (Test-Path -LiteralPath $venv)' in SCRIPT


def test_first_run_does_not_call_winget_with_valid_windows_python():
    assert 'if (-not $systemPy) {' in SCRIPT
    assert "InstallWinget 'Python.Python.3.11' 'Python 3.11'" in SCRIPT
    assert 'if (-not $creationPy) {' in SCRIPT


def test_readonly_wsl_mode_can_use_project_314_python():
    assert 'if ($CheckWSLOnly) {' in SCRIPT
    assert '$py = FindWindowsPython' in SCRIPT
    assert 'WSL discovery' in SCRIPT


def test_exit_status_probe_avoids_windows_powershell_quote_loss():
    assert 'sys.exit(0 if (3,10) <= sys.version_info[:2]' in SCRIPT
    assert "-c $probeCode" in SCRIPT
    assert 'if ($probeExit -eq 0)' in SCRIPT
    assert 'if ($null -eq $wingetExitCode)' in SCRIPT
