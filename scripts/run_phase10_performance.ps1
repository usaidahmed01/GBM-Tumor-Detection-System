param(
    [switch]$FrontendBuild
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PreviousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
Push-Location $ProjectRoot
try {
    $ArgsList = @("-m", "gbm_ai.validation.performance")
    if ($FrontendBuild) { $ArgsList += "--frontend-build" }

    Write-Host "Running NeuroGlioma AI Phase 10 engineering performance smoke..." -ForegroundColor Cyan
    & python @ArgsList
    if ($LASTEXITCODE -ne 0) {
        throw "Phase 10 performance smoke failed with exit code $LASTEXITCODE"
    }
}
finally {
    $env:PYTHONPATH = $PreviousPythonPath
    Pop-Location
}
