# NeuroGlioma AI — Free-Tier-First Deployment Plan

This deployment plan is for the university/demo release. It is deliberately **free-tier-first** and uses no paid model API.
It must not be described as a clinical production deployment.

## Target hosted architecture

| Layer | Free-tier-first target | Purpose |
|---|---|---|
| Web UI | Vercel Hobby | Next.js frontend |
| API container | Oracle Cloud Always Free Ampere A1 VM + Docker | FastAPI + CPU AI runtime |
| PostgreSQL | Neon Free | Application/audit database |
| MRI/model runtime storage | Oracle Always Free block volume initially | Persistent private runtime assets on the VM; never Git |
| 2D classifier weights | Runtime-managed private assets | Never commit weights to Git |
| MONAI bundle / atlas | Runtime-managed cache/assets | Never commit downloaded model/atlas assets to Git |

## Important zero-cost boundary

The goal is **$0 for low-volume university/demo use**, not an unlimited-cost guarantee. Usage-based services can exceed their free allowance if traffic or compute grows. Configure scale-to-zero, a maximum of one API instance, quotas/budgets, and monitor usage.

The current 3D SegResNet path is CPU/memory intensive. A free hosted environment may be suitable for occasional demonstrations but must not be represented as a guaranteed-performance production inference service. If the free compute ceiling is insufficient, the system must return a controlled unavailable/indeterminate path rather than silently changing the model or using a paid API.

## Deployment sequence

### 1. Database — Neon Free

1. Create one PostgreSQL project.
2. Copy its connection string.
3. Convert it to the project's SQLAlchemy form when needed, for example:

   `postgresql+psycopg://USER:PASSWORD@HOST/DB?sslmode=require`

4. Set `GBM_DATABASE_URL` only as a secret/environment variable.
5. Run `alembic upgrade head` against the deployment database before public testing.

### 2. Backend — Oracle Cloud Always Free Ampere A1 + Docker

The root `Dockerfile` now has an `oracle-arm64-preflight` target and the normal `api` runtime target. Before creating the Oracle VM, run `scripts/run_oracle_arm64_preflight.ps1` to prove the full Python/native dependency stack builds for `linux/arm64`.

Frozen free-account policy:

- `VM.Standard.A1.Flex` only
- conservative Always Free allocation: at most `2 OCPUs / 12 GB RAM` total
- CPU inference only
- do not upgrade the Oracle account to Pay As You Go for this project
- environment: `production`
- debug: `false`
- database URL: secret environment variable
- model/runtime assets: persistent runtime volume, never Git or the container image

Do not place real patient data into an academic public deployment.

### 3. Frontend — Vercel Hobby

Import the GitHub repository and choose `frontend` as the Vercel project root.
Set:

`GBM_BACKEND_ORIGIN=https://YOUR-ORACLE-BACKEND-HOST`

The browser continues to call the Next.js `/gbm-api/*` proxy rather than embedding the backend URL throughout the UI.

### 4. Runtime MRI/model storage — Oracle persistent volume

For the initial university/demo deployment, keep private model bundles, atlas files and temporary/protected MRI objects on persistent Oracle block storage attached to the A1 VM. They remain outside Git and outside the Docker image.

A later S3-compatible object-store adapter can still be added if needed, but it is not required to prove the first free Docker deployment. Hosted persistence must be validated before claiming durable study retention.

Until hosted persistence has been validated, this remains **demonstration-only**. Do **not** claim durable cloud MRI storage.

## Free deployment acceptance gate

A public demo is deployment-ready only when all of these are true:

- `npm run build` passes.
- Phase 10 automated validation passes.
- Alembic reports exactly one current head.
- Git contains no MRI data, `.env`, weights, `.next`, `node_modules`, or runtime storage.
- Cloud environment secrets are configured outside Git.
- Missing classifier/model assets remain visibly unavailable rather than being substituted.
- No clinical-validation or regulatory claim is displayed.

## What "free deployment" means for this project

We will prefer free tiers and free/open-source models throughout. We will not introduce a paid LLM/model API merely to make deployment easier. If a future provider removes its free tier, replace the hosting layer rather than silently changing the clinical/AI methodology.
