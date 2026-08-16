param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
)

$ErrorActionPreference = 'Stop'
$target = Join-Path $ProjectRoot 'var\models\classifier\efficientnetv2s_seed42'
New-Item -ItemType Directory -Force -Path $target | Out-Null

$expected = [ordered]@{
    'efficientnetv2s_fold0_best_model.pt' = 'e996bea99a7b1e2ca60e96fc8b5dc783afc0d6671325bb83486d458819f2520c'
    'efficientnetv2s_fold1_best_model.pt' = '3ed6b41b22056898ef4c4c48daa5dd4368a9389ef4ed5d831573f027f9b62e08'
    'efficientnetv2s_fold2_best_model.pt' = '1ab1059a0e1f5bbf00123c1059ffa81cdd876cf0e7f95b53183fd23d29a1d247'
    'efficientnetv2s_fold3_best_model.pt' = '6c25e266adf89e0a1bee72dbbb0392a456e4ec1235969d8c9fce1d94c6aa5a91'
    'efficientnetv2s_fold4_best_model.pt' = '9f4952120de8e202c2b795f41f7dc96999bec1046b6dee5e9b76eba59c81b106'
}

Write-Host 'NeuroGlioma AI - classifier runtime asset installer'
Write-Host 'Searching local project checkpoints by frozen SHA-256. No arbitrary fold selection is used.'

$skipPattern = '\\.git\\|\\.venv\\|\\node_modules\\|\\.next\\|\\var\\models\\classifier\\efficientnetv2s_seed42\\'
$candidates = Get-ChildItem -Path $ProjectRoot -Recurse -File -Include *.pt,*.pth -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notmatch $skipPattern }

$hashToPath = @{}
foreach ($file in $candidates) {
    try {
        $hash = (Get-FileHash -Algorithm SHA256 -Path $file.FullName).Hash.ToLowerInvariant()
        if (-not $hashToPath.ContainsKey($hash)) { $hashToPath[$hash] = $file.FullName }
    } catch {}
}

$installed = 0
foreach ($name in $expected.Keys) {
    $hash = $expected[$name]
    $dest = Join-Path $target $name
    if (Test-Path $dest) {
        $current = (Get-FileHash -Algorithm SHA256 -Path $dest).Hash.ToLowerInvariant()
        if ($current -eq $hash) {
            Write-Host "[OK] $name already installed"
            $installed++
            continue
        }
        Write-Warning "$name exists but checksum does not match the frozen artifact; leaving it untouched."
        continue
    }
    if ($hashToPath.ContainsKey($hash)) {
        Copy-Item -LiteralPath $hashToPath[$hash] -Destination $dest
        Write-Host "[OK] Installed $name from $($hashToPath[$hash])"
        $installed++
    } else {
        Write-Warning "Could not find frozen checkpoint for $name"
    }
}

Write-Host "Classifier checkpoints ready: $installed / $($expected.Count)"
if ($installed -ne $expected.Count) {
    Write-Host 'Keep the classifier blocked until all five exact frozen checkpoints are available.'
    exit 2
}
Write-Host 'Classifier runtime assets are ready.'
