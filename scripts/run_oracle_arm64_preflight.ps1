param(
    [string]$ImageName = "neuroglioma-api:oracle-arm64-preflight",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$reportDir = Join-Path $projectRoot "var\validation"
$reportPath = Join-Path $reportDir "phase10_step5a_oracle_arm64_preflight.txt"

New-Item -ItemType Directory -Force -Path $reportDir | Out-Null
Set-Location $projectRoot

function Invoke-DockerChecked {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$DockerArgs
    )

    & docker @DockerArgs
    if ($LASTEXITCODE -ne 0) {
        $joined = $DockerArgs -join " "
        throw "Docker command failed: docker $joined"
    }
}

function Invoke-DockerCaptured {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$DockerArgs
    )

    # Windows PowerShell 5.1 turns native STDERR into ErrorRecord objects.
    # QEMU/PyTorch can emit benign ARM CPU-probing messages on STDERR even when
    # the container exits with code 0. Temporarily keep native STDERR non-fatal
    # and decide success strictly from the Docker exit code + PASS marker.
    $savedPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $output = & docker @DockerArgs 2>&1
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $savedPreference
    }

    return [PSCustomObject]@{
        Output = @($output)
        ExitCode = $exitCode
    }
}

Write-Host "NeuroGlioma AI - Oracle Ampere A1 ARM64 Docker preflight" -ForegroundColor Cyan
Write-Host "Native ARM64 compatibility is tested through Docker Buildx/QEMU before Oracle VM creation." -ForegroundColor DarkGray

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker CLI was not found. Install/start Docker Desktop, then retry."
}

& docker info *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Docker Desktop/engine is not running. Start Docker Desktop and retry."
}

Write-Host ""
Write-Host "Checking Docker Buildx..." -ForegroundColor Yellow
Invoke-DockerChecked -DockerArgs @("buildx", "version")
Invoke-DockerChecked -DockerArgs @("buildx", "inspect", "--bootstrap")

if ($SkipBuild) {
    Write-Host ""
    Write-Host "[1/3] Reusing the existing ARM64 preflight image..." -ForegroundColor Yellow
    & docker image inspect $ImageName *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "-SkipBuild was requested, but Docker image '$ImageName' does not exist. Rerun without -SkipBuild."
    }
}
else {
    Write-Host ""
    Write-Host "[1/3] Building the backend dependency stack for linux/arm64..." -ForegroundColor Yellow
    Invoke-DockerChecked -DockerArgs @(
        "buildx", "build",
        "--platform", "linux/arm64",
        "--target", "oracle-arm64-preflight",
        "--tag", $ImageName,
        "--load",
        "--progress", "plain",
        "."
    )
}

Write-Host ""
Write-Host "[2/3] Confirming the resulting image architecture..." -ForegroundColor Yellow
$inspectResult = Invoke-DockerCaptured -DockerArgs @("image", "inspect", $ImageName, "--format", "{{.Architecture}}")
if ($inspectResult.ExitCode -ne 0) {
    $inspectResult.Output | ForEach-Object { Write-Host $_ }
    throw "Unable to inspect the ARM64 preflight image."
}
$architecture = ($inspectResult.Output | Out-String).Trim()
if ($architecture -ne "arm64") {
    throw "Expected Docker image architecture arm64, got: $architecture"
}
Write-Host "Image architecture: arm64" -ForegroundColor Green

Write-Host ""
Write-Host "[3/3] Running the ARM64 runtime/model-construction smoke test..." -ForegroundColor Yellow
$runtimeResult = Invoke-DockerCaptured -DockerArgs @("run", "--rm", "--platform", "linux/arm64", $ImageName)
$runtimeResult.Output | ForEach-Object { Write-Host $_ }
$runtimeResult.Output | Out-File -FilePath $reportPath -Encoding utf8

if ($runtimeResult.ExitCode -ne 0) {
    throw "ARM64 runtime smoke test failed with Docker exit code $($runtimeResult.ExitCode). See report: $reportPath"
}

$runtimeText = $runtimeResult.Output | Out-String
if ($runtimeText -notmatch "ARM64 Docker compatibility:\s+PASS") {
    throw "The preflight container exited successfully but did not report the expected PASS marker."
}

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "ORACLE ARM64 DOCKER PREFLIGHT: PASS" -ForegroundColor Green
Write-Host "Report: $reportPath"
Write-Host "The MIDR_EL1 message seen under local ARM64 emulation is treated as diagnostic STDERR, not as a failed container run."
Write-Host "Next step: Oracle Always Free account and Ampere A1 VM setup."
