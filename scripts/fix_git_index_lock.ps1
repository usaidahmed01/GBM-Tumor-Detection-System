\
param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$gitDir = Join-Path $repoRoot ".git"
$lockPath = Join-Path $gitDir "index.lock"

if (-not (Test-Path $gitDir)) {
    throw "This folder is not a Git repository: $repoRoot"
}

$gitProcesses = @(Get-Process -Name git,git-remote-https,git-remote-http -ErrorAction SilentlyContinue)
if ($gitProcesses.Count -gt 0 -and -not $Force) {
    Write-Host "An active Git process is still running. Close any Git commit/editor/source-control operation first." -ForegroundColor Yellow
    $gitProcesses | Select-Object Id, ProcessName, StartTime | Format-Table -AutoSize
    Write-Host "Then rerun this script. Use -Force only if you have confirmed the processes are stale." -ForegroundColor Yellow
    exit 2
}

if ($gitProcesses.Count -gt 0 -and $Force) {
    Write-Host "Force mode requested. Stopping stale Git processes..." -ForegroundColor Yellow
    $gitProcesses | Stop-Process -Force
    Start-Sleep -Milliseconds 300
}

if (Test-Path $lockPath) {
    Remove-Item $lockPath -Force
    Write-Host "Removed stale Git index lock: $lockPath" -ForegroundColor Green
} else {
    Write-Host "No Git index lock exists. Repository lock state is already clean." -ForegroundColor Green
}

Write-Host "Checking repository..." -ForegroundColor Cyan
git status --short
if ($LASTEXITCODE -ne 0) {
    throw "git status failed after lock cleanup"
}

Write-Host "Git index is available. You can run: git add ." -ForegroundColor Green
