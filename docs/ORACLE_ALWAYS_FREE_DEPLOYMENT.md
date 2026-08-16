# NeuroGlioma AI — Oracle Always Free Deployment Track

This document freezes the free-hosting direction for the university/demo release before any Oracle resources are created.

## Cost boundary

The deployment must stay on **Always Free-eligible resources only**. Do not upgrade the Oracle account to Pay As You Go for this project. Do not select a VM shape, boot image, block volume, public IP option, load balancer, database or other resource unless the Oracle Console marks the intended choice as Always Free-eligible or the step explicitly verifies it against the current Oracle documentation.

For the conservative post-trial free-account plan, reserve at most:

- `VM.Standard.A1.Flex`
- **2 OCPUs total**
- **12 GB RAM total**
- ARM64 / AArch64 Linux

The VM will run the existing FastAPI + PyTorch + MONAI backend as a Docker container. No paid AI/model API is introduced.

## Why the ARM64 preflight comes first

The Oracle A1 free compute shape is ARM-based. The local development computer is x86-64 Windows, so a normal local Docker build is not enough evidence that all native Python wheels work on AArch64. Before account/VM setup, run the repository's Buildx preflight for `linux/arm64`.

The preflight verifies:

1. the complete cumulative `requirements.txt` can install in the ARM64 Python 3.11 image;
2. `pip check` is clean;
3. PyTorch / TorchVision / MONAI load on ARM64 CPU;
4. SimpleITK and NiBabel native imaging operations work;
5. EfficientNetV2-S constructs without downloading weights;
6. the frozen SegResNet architecture constructs without downloading weights;
7. the real FastAPI application module imports;
8. the resulting image architecture is actually `arm64`.

It deliberately does **not** embed or download private runtime assets. The five classifier checkpoints, MONAI bundle, localization atlas, uploaded MRI studies and secrets remain outside Git and outside the image.

## Local Windows preflight

From the repository root with Docker Desktop running:

```powershell
.\scripts\run_oracle_arm64_preflight.ps1
```

Expected final marker:

```text
ORACLE ARM64 DOCKER PREFLIGHT: PASS
```

The detailed result is written to:

```text
var\validation\phase10_step5a_oracle_arm64_preflight.txt
```

Do not create the Oracle VM until this check passes. If the ARM64 dependency build fails, fix that compatibility issue first rather than changing the frozen AI models or paying for a different platform.

## What happens after the preflight passes

The next deployment step will be guided screen-by-screen:

1. create/sign in to the Oracle Cloud Free Tier account;
2. choose the home region carefully;
3. create a dedicated `NeuroGliomaAI` compartment;
4. create the VCN/public subnet;
5. create one Always Free Ampere A1 instance;
6. allocate exactly 2 OCPUs / 12 GB RAM (or less if the Console's current free allowance is lower);
7. choose an Always Free-eligible Ubuntu ARM64 image;
8. create/download the SSH key safely;
9. confirm the cost estimate is `$0` / Always Free before pressing Create;
10. connect over SSH and install Docker Engine;
11. deploy the tested ARM64 backend container;
12. add runtime assets on persistent storage without committing them to Git;
13. connect Neon PostgreSQL and run Alembic migrations;
14. configure GitHub-based automatic redeployment;
15. deploy the Vercel frontend and run the complete self-verification matrix.

## Remaining resource risks

- Always Free A1 capacity can be temporarily unavailable in a selected home region.
- Oracle may reclaim compute it classifies as idle under its Always Free policy.
- A successful architecture/dependency preflight does not prove that a full 3D MRI inference fits the final VM RAM budget; that is measured on the actual A1 VM before public testing.
- This remains a university/demo deployment and must not be represented as clinically validated or cleared for diagnostic use.
