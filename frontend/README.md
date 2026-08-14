# GBM CDSS Clinical Viewer — Phase 8 Step 2

This Next.js/React UI is the read-only clinical MRI review workspace for compatible volumetric studies that already completed the Phase 6 segmentation pipeline.

## Current Step 2 capabilities

- Cornerstone3D NIfTI rendering of the protected model-space T1C/T1/T2/FLAIR volumes.
- Axial, coronal and sagittal MPR viewports.
- BraTS combined labelmap overlay (WT/TC/ET), visibility and opacity controls.
- Window/level, pan, zoom, mouse-wheel slice navigation and camera reset.
- Physical quantification and atlas-localization summaries from the backend manifest.
- Research/safety notices and current segmentation review status.
- Responsive medical-style UI with reduced-motion support.

Manual brush/erase editing and accept/reject actions are intentionally not enabled until Phase 8 Step 3, where persistence, audit history and downstream volume/location recalculation can be implemented together.

## Local run

1. Start the FastAPI backend on `http://127.0.0.1:8000`.
2. Copy `.env.local.example` to `.env.local` if you need to change the backend origin.
3. Run `npm install`.
4. Run `npm run dev`.
5. Open `http://localhost:3000` and enter a compatible study UUID.

The browser uses `/gbm-api/*`; Next.js proxies that path to the FastAPI `/api/v1/*` routes, so protected object paths remain server-side.
