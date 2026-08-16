$ErrorActionPreference = "Stop"

Write-Host "NeuroGlioma AI — Git generated-artifact cleanup" -ForegroundColor Cyan
Write-Host "Stop npm run dev before running this script so frontend/.next/dev/lock is released." -ForegroundColor Yellow

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

# Remove generated frontend trees from Git's index if they were added/tracked earlier.
git rm -r --cached --ignore-unmatch frontend/.next frontend/node_modules frontend/out frontend/.turbo frontend/.cache | Out-Host

# Stage the hygiene rules first.
git add .gitignore .gitattributes

Write-Host ""
Write-Host "Ignored path check:" -ForegroundColor Cyan
git check-ignore -v frontend/.next/dev/lock 2>$null | Out-Host

$trackedNext = git ls-files frontend/.next
if ($trackedNext) {
    Write-Host "ERROR: frontend/.next still has tracked files:" -ForegroundColor Red
    $trackedNext | Out-Host
    exit 1
}

$trackedModules = git ls-files frontend/node_modules
if ($trackedModules) {
    Write-Host "ERROR: frontend/node_modules still has tracked files." -ForegroundColor Red
    exit 1
}

Write-Host "Generated frontend artifacts are no longer tracked." -ForegroundColor Green
Write-Host "You can now run: git add ." -ForegroundColor Green
