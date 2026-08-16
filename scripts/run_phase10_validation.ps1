param(
    [switch]$Full,
    [switch]$FrontendBuild
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot
try {
    $ArgsList = @("-m", "gbm_ai.validation.runner")
    if ($Full) { $ArgsList += "--full" }
    if ($FrontendBuild) { $ArgsList += "--frontend-build" }

    Write-Host "Running NeuroGlioma AI Phase 10 validation matrix..." -ForegroundColor Cyan
    & python @ArgsList
    if ($LASTEXITCODE -ne 0) {
        throw "Phase 10 validation matrix failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
