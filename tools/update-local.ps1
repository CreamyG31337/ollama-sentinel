<#
.SYNOPSIS
  Apply local source changes to the installed/running ollama-sentinel.

.DESCRIPTION
  The package is installed with `pip install -e`, so ordinary .py edits are live
  immediately and the 15-minute --once task picks them up on its next run with no
  action at all. Two things do NOT self-update:

    1. the long-running GUI process (`--gui` / `--gui --start-minimized`), which holds old code in memory
    2. dependencies / entry points, when pyproject.toml changes

  This script handles both, then verifies with the test suite.

.EXAMPLE
  pwsh -File tools\update-local.ps1
  pwsh -File tools\update-local.ps1 -Reinstall   # force the pip step
#>
[CmdletBinding()]
param(
    [switch]$Reinstall,      # force pip install even if pyproject looks unchanged
    [switch]$SkipTests
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$py = 'C:\Users\cream\AppData\Local\Programs\Python\Python312\python.exe'
$stamp = Join-Path $root '.pyproject.sha256'
$trayTask = 'ollama-sentinel-tray'

Write-Host "ollama-sentinel: applying local changes from $root" -ForegroundColor Cyan

# --- 1. reinstall only if pyproject.toml actually changed -------------------
$pyproject = Join-Path $root 'pyproject.toml'
$hash = (Get-FileHash $pyproject -Algorithm SHA256).Hash
$prev = if (Test-Path $stamp) { (Get-Content $stamp -Raw).Trim() } else { '' }

if ($Reinstall -or $hash -ne $prev) {
    if ($hash -ne $prev -and $prev) {
        Write-Host "  pyproject.toml changed -- reinstalling" -ForegroundColor Yellow
    } else {
        Write-Host "  reinstalling" -ForegroundColor Yellow
    }
    Push-Location $root
    try   { & $py -m pip install -e ".[windows,gui]" --quiet }
    finally { Pop-Location }
    Set-Content -Path $stamp -Value $hash -NoNewline
} else {
    Write-Host "  deps unchanged -- editable install already live" -ForegroundColor DarkGray
}

# --- 2. restart the GUI/tray task so it picks up new code -------------------
# Logon task should use: pythonw.exe -m ollama_sentinel --gui --start-minimized
$task = Get-ScheduledTask -TaskName $trayTask -ErrorAction SilentlyContinue
if ($task) {
    $info = $task | Get-ScheduledTaskInfo
    # 267009 = 0x41301 = currently running
    if ($info.LastTaskResult -eq 267009) {
        Write-Host "  restarting tray" -ForegroundColor Yellow
        Stop-ScheduledTask -TaskName $trayTask
        # Stop-ScheduledTask kills the task, but the flet child can outlive it.
        Get-Process flet, pythonw -ErrorAction SilentlyContinue |
            Where-Object { $_.MainWindowTitle -eq 'ollama-sentinel' -or $_.Name -eq 'pythonw' } |
            Stop-Process -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
        Start-ScheduledTask -TaskName $trayTask
    } else {
        Write-Host "  tray not running -- nothing to restart" -ForegroundColor DarkGray
    }
} else {
    Write-Host "  tray task not registered" -ForegroundColor DarkGray
}

# --- 3. verify --------------------------------------------------------------
if (-not $SkipTests) {
    Push-Location $root
    try { & $py -m pytest tests/ -q }
    finally { Pop-Location }
}

Write-Host "  smoke test:" -ForegroundColor Cyan
& ollama-sentinel --once
Write-Host "done. The --once alarm task needs no restart; it re-reads code every run." -ForegroundColor Green
