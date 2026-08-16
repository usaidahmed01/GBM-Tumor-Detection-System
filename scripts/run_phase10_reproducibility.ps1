param(
    [switch]$DeepPythonInstall,
    [switch]$DeepFrontendInstall
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PreviousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$CleanVenv = Join-Path $ProjectRoot ".phase10-clean-venv"

Push-Location $ProjectRoot
try {
    Write-Host "Checking NeuroGlioma AI reproducibility prerequisites..." -ForegroundColor Cyan
    & python -m gbm_ai.validation.reproducibility
    if ($LASTEXITCODE -ne 0) {
        throw "Reproducibility prerequisite check failed. Fix the reported item before deep reinstall tests."
    }

    if ($DeepPythonInstall) {
        Write-Host "Creating clean Python environment..." -ForegroundColor Cyan
        if (Test-Path $CleanVenv) {
            Remove-Item -Recurse -Force $CleanVenv
        }
        & python -m venv $CleanVenv
        $CleanPython = Join-Path $CleanVenv "Scripts\python.exe"
        & $CleanPython -m pip install --upgrade pip
        & $CleanPython -m pip install -r requirements.txt
        & $CleanPython -m pip check
        & $CleanPython -m pytest -q `
            tests/test_phase10_validation_matrix.py `
            tests/test_phase10_validation_runner.py `
            tests/test_phase10_performance_reproducibility.py `
            tests/test_phase4_backend_foundation.py
        if ($LASTEXITCODE -ne 0) {
            throw "Clean Python environment smoke tests failed."
        }
    }

    if ($DeepFrontendInstall) {
        if (!(Test-Path "frontend\package-lock.json")) {
            throw "frontend/package-lock.json is required before npm ci reproducibility testing."
        }
        Push-Location "frontend"
        try {
            Write-Host "Reinstalling frontend strictly from package-lock.json..." -ForegroundColor Cyan
            & npm ci
            if ($LASTEXITCODE -ne 0) { throw "npm ci failed" }
            & npm run build
            if ($LASTEXITCODE -ne 0) { throw "frontend production build failed" }
        }
        finally {
            Pop-Location
        }
    }

    Write-Host "Phase 10 Step 3 reproducibility workflow completed." -ForegroundColor Green
}
finally {
    $env:PYTHONPATH = $PreviousPythonPath
    Pop-Location
}
