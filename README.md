# NeuroGlioma AI

NeuroGlioma AI is the product interface for the project **AI-Based Clinical Decision Support System for Early Identification of Glioblastoma Multiforme**.

## Local development — Windows / PowerShell

### Backend

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
alembic upgrade head
python -m uvicorn gbm_ai.api.main:app --reload --host 127.0.0.1 --port 8000
```

### Segmentation worker

```powershell
.\.venv\Scripts\Activate.ps1
python -m gbm_ai.api.workers.segmentation_worker
```

### Frontend

```powershell
cd frontend
Copy-Item .env.local.example .env.local
npm ci
npm run dev
```

Open `http://localhost:3000`.

## Runtime assets kept outside Git

The following are environment-managed and intentionally excluded from source control:

- EfficientNetV2-S classifier checkpoints under `var/models/classifier/efficientnetv2s_seed42/`;
- frozen MONAI segmentation bundle under `var/model_bundles/`;
- localization template/atlas assets under `var/localization_atlas/`;
- uploaded MRI and derived case artifacts under protected local storage.

The application must not silently substitute missing model assets.

## Validation

Full automated validation plus the frontend production build:

```powershell
.\scripts\run_phase10_validation.ps1 -Full -FrontendBuild
```

Engineering performance smoke:

```powershell
.\scripts\run_phase10_performance.ps1 -FrontendBuild
```

Reproducibility prerequisite check:

```powershell
.\scripts\run_phase10_reproducibility.ps1
```

Optional deeper clean-install checks:

```powershell
.\scripts\run_phase10_reproducibility.ps1 -DeepPythonInstall -DeepFrontendInstall
```

These are software/engineering validation workflows. They do not establish clinical validation or regulatory clearance.

## Free-tier-first deployment

The release is prepared around free/open-source model execution and free-tier-first hosting. See `docs/FREE_DEPLOYMENT.md` for the current deployment blueprint and its resource/clinical limitations. The project does not require a paid model API.

Release verification:

```powershell
python -m gbm_ai.api.verify_phase10_release
```

If Git reports a stale `.git/index.lock`, first make sure no Git operation is active and then run:

```powershell
.\scripts\fix_git_index_lock.ps1
```
