# NeuroGlioma AI — Free-Tier-First Deployment Plan

This deployment plan is for the university/demo release. It is deliberately **free-tier-first** and uses no paid model API.
It must not be described as a clinical production deployment.

## Target hosted architecture

| Layer | Free-tier-first target | Purpose |
|---|---|---|
| Web UI | Vercel Hobby | Next.js frontend |
| API container | Google Cloud Run, scale-to-zero | FastAPI backend |
| PostgreSQL | Neon Free | Application/audit database |
| MRI object storage | Cloudflare R2 free monthly allowance | Protected study/derived objects once the remote object-store adapter is enabled |
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

### 2. Backend — Google Cloud Run

The root `Dockerfile` builds the FastAPI API only. It deliberately excludes frontend build output, patient/MRI data, local model caches and local secrets.

Recommended demo settings:

- minimum instances: `0`
- maximum instances: `1`
- request authentication/public access: choose according to demo needs
- environment: `production`
- debug: `false`
- database URL: secret environment variable
- model/runtime assets: mounted or downloaded by an explicit deployment procedure, never from Git

Do not place real patient data into an academic public deployment.

### 3. Frontend — Vercel Hobby

Import the GitHub repository and choose `frontend` as the Vercel project root.
Set:

`GBM_BACKEND_ORIGIN=https://YOUR-CLOUD-RUN-SERVICE`

The browser continues to call the Next.js `/gbm-api/*` proxy rather than embedding the backend URL throughout the UI.

### 4. Object storage — Cloudflare R2

R2 is the preferred zero-cost-capable object-storage target because the project must not rely on an ephemeral container filesystem for durable MRI objects.

The current source release still defaults to the protected local object-store implementation. Do **not** claim durable cloud MRI storage until the R2/S3 adapter has been enabled and validated. Until then, hosted API deployments are demonstration-only and uploaded objects can be treated as ephemeral.

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
