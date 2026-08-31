<#
.SYNOPSIS
  Create a Desktop shortcut that opens the ollama-sentinel window.

.DESCRIPTION
  Uses pythonw.exe so no console window appears, and --gui rather than --tray
  so it does not add a second tray icon when the logon tray task is already
  running.

.EXAMPLE
  pwsh -File tools\install-shortcut.ps1
  pwsh -File tools\install-shortcut.ps1 -Remove
#>
[CmdletBinding()]
param(
    [string]$Name = 'Ollama Sentinel',
    [switch]$Remove
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$desktop = [Environment]::GetFolderPath('Desktop')
$lnkPath = Join-Path $desktop "$Name.lnk"

if ($Remove) {
    if (Test-Path $lnkPath) { Remove-Item $lnkPath -Force; "removed $lnkPath" }
    else { "no shortcut at $lnkPath" }
    return
}

# Prefer the pythonw next to whichever python is on PATH.
$python = (Get-Command python -ErrorAction SilentlyContinue).Source
$pyw = if ($python) { Join-Path (Split-Path $python) 'pythonw.exe' } else { $null }
if (-not $pyw -or -not (Test-Path $pyw)) {
    throw "pythonw.exe not found next to '$python'. Pass the right interpreter or install Python for Windows."
}

$ico = Join-Path $root 'assets\ollama-sentinel.ico'

$ws = New-Object -ComObject WScript.Shell
$lnk = $ws.CreateShortcut($lnkPath)
$lnk.TargetPath       = $pyw
$lnk.Arguments        = '-m ollama_sentinel --gui'
$lnk.WorkingDirectory = $root
$lnk.Description      = 'Ollama GPU spill / paging / VRAM monitor'
$lnk.WindowStyle      = 1
if (Test-Path $ico) { $lnk.IconLocation = "$ico,0" }
$lnk.Save()

"created $lnkPath"
"  $pyw -m ollama_sentinel --gui  (cwd: $root)"
