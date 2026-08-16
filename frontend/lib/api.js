export const VIEWER_UI_VERSION = 'phase8_step3_clinician_mask_review_v1';

export function proxyApiUrl(apiUrl) {
  if (!apiUrl) return null;
  return apiUrl.replace(/^\/api\/v1(?=\/|$)/, '/gbm-api');
}

async function readApiError(response) {
  try {
    const payload = await response.json();
    const detail = payload?.detail;
    if (typeof detail === 'string') return detail;
    if (detail?.message) return detail.message;
  } catch {
    // Response was not JSON. Keep the safe generic message below.
  }
  return `Request failed (${response.status})`;
}

async function checkedJson(response) {
  if (!response.ok) throw new Error(await readApiError(response));
  return response.json();
}

export async function fetchViewerManifest(studyUuid, { signal } = {}) {
  return checkedJson(await fetch(
    `/gbm-api/studies/${encodeURIComponent(studyUuid)}/viewer/manifest`,
    {
      method: 'GET',
      cache: 'no-store',
      signal,
      headers: { Accept: 'application/json' },
    },
  ));
}

export async function submitSegmentationReview(studyUuid, action, note = '') {
  return checkedJson(await fetch(
    `/gbm-api/studies/${encodeURIComponent(studyUuid)}/viewer/review`,
    {
      method: 'POST',
      cache: 'no-store',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({ action, note: note.trim() || null }),
    },
  ));
}

export async function submitLabelmapCorrection(
  studyUuid,
  { rawLabelmap, sourceChecksumSha256, note = '' },
) {
  const form = new FormData();
  form.append('source_checksum_sha256', sourceChecksumSha256);
  if (note.trim()) form.append('note', note.trim());
  form.append(
    'labelmap',
    new Blob([rawLabelmap], { type: 'application/octet-stream' }),
    'clinician_labelmap_uint8.bin',
  );
  return checkedJson(await fetch(
    `/gbm-api/studies/${encodeURIComponent(studyUuid)}/viewer/corrections`,
    { method: 'POST', cache: 'no-store', body: form, headers: { Accept: 'application/json' } },
  ));
}

export async function fetchReviewHistory(studyUuid) {
  return checkedJson(await fetch(
    `/gbm-api/studies/${encodeURIComponent(studyUuid)}/viewer/review/history`,
    { method: 'GET', cache: 'no-store', headers: { Accept: 'application/json' } },
  ));
}

export async function createUnifiedIntake(payload) {
  return checkedJson(await fetch('/gbm-api/intake/studies', {
    method: 'POST',
    cache: 'no-store',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify(payload),
  }));
}

export async function uploadStudySource(studyUuid, file) {
  const form = new FormData();
  form.append('file', file);
  return checkedJson(await fetch(`/gbm-api/studies/${encodeURIComponent(studyUuid)}/upload`, {
    method: 'POST',
    cache: 'no-store',
    headers: { Accept: 'application/json' },
    body: form,
  }));
}

export async function runStudyQc(studyUuid) {
  return checkedJson(await fetch(`/gbm-api/studies/${encodeURIComponent(studyUuid)}/qc`, {
    method: 'POST', cache: 'no-store', headers: { Accept: 'application/json' },
  }));
}

export async function confirmBrainScope(studyUuid, isBrainMri = true) {
  return checkedJson(await fetch(`/gbm-api/studies/${encodeURIComponent(studyUuid)}/brain-scope-confirmation`, {
    method: 'PUT', cache: 'no-store',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify({ is_brain_mri: isBrainMri }),
  }));
}

export async function confirmNiftiSequenceMapping(studyUuid, mapping) {
  return checkedJson(await fetch(`/gbm-api/studies/${encodeURIComponent(studyUuid)}/nifti-sequence-mapping`, {
    method: 'PUT', cache: 'no-store',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify(mapping),
  }));
}

export async function fetchStudySeries(studyUuid) {
  return checkedJson(await fetch(`/gbm-api/studies/${encodeURIComponent(studyUuid)}/series`, {
    method: 'GET', cache: 'no-store', headers: { Accept: 'application/json' },
  }));
}

export async function confirmDicomSeriesSequence(seriesUuid, sequence) {
  return checkedJson(await fetch(`/gbm-api/series/${encodeURIComponent(seriesUuid)}/sequence-confirmation`, {
    method: 'PUT', cache: 'no-store',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify({ sequence }),
  }));
}

export async function routeStudyCapabilities(studyUuid) {
  return checkedJson(await fetch(`/gbm-api/studies/${encodeURIComponent(studyUuid)}/capabilities/route`, {
    method: 'POST', cache: 'no-store', headers: { Accept: 'application/json' },
  }));
}

async function postNoBody(path) {
  return checkedJson(await fetch(path, { method: 'POST', cache: 'no-store', headers: { Accept: 'application/json' } }));
}

export const runSegmentationPreflight = (studyUuid) => postNoBody(`/gbm-api/studies/${encodeURIComponent(studyUuid)}/segmentation/preflight`);
export const prepareSegmentationVolumes = (studyUuid) => postNoBody(`/gbm-api/studies/${encodeURIComponent(studyUuid)}/segmentation/prepare-volumes`);
export const prepareSegmentationGeometry = (studyUuid) => postNoBody(`/gbm-api/studies/${encodeURIComponent(studyUuid)}/segmentation/prepare-model-geometry`);
export const prepareSegmentationModelInput = (studyUuid) => postNoBody(`/gbm-api/studies/${encodeURIComponent(studyUuid)}/segmentation/prepare-model-input`);
export const enqueueSegmentationJob = (studyUuid) => postNoBody(`/gbm-api/studies/${encodeURIComponent(studyUuid)}/segmentation/jobs`);
export const runTumorQuantification = (studyUuid) => postNoBody(`/gbm-api/studies/${encodeURIComponent(studyUuid)}/quantification/run`);
export const runAnatomicalLocalization = (studyUuid) => postNoBody(`/gbm-api/studies/${encodeURIComponent(studyUuid)}/localization/run`);

export async function fetchSegmentationJob(jobUuid) {
  return checkedJson(await fetch(`/gbm-api/segmentation/jobs/${encodeURIComponent(jobUuid)}`, {
    method: 'GET', cache: 'no-store', headers: { Accept: 'application/json' },
  }));
}

export async function fuseStudyDecision(studyUuid) {
  return checkedJson(await fetch(`/gbm-api/studies/${encodeURIComponent(studyUuid)}/decision/fuse`, {
    method: 'POST', cache: 'no-store', headers: { Accept: 'application/json' },
  }));
}

export async function fetchReportPreview(studyUuid) {
  return checkedJson(await fetch(`/gbm-api/studies/${encodeURIComponent(studyUuid)}/report/preview`, {
    method: 'GET', cache: 'no-store', headers: { Accept: 'application/json' },
  }));
}

export async function finalizeClinicalReport(studyUuid, payload) {
  return checkedJson(await fetch(`/gbm-api/studies/${encodeURIComponent(studyUuid)}/report/finalize`, {
    method: 'POST', cache: 'no-store',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify(payload),
  }));
}

export async function fetchCurrentReport(studyUuid) {
  return checkedJson(await fetch(`/gbm-api/studies/${encodeURIComponent(studyUuid)}/report/current`, {
    method: 'GET', cache: 'no-store', headers: { Accept: 'application/json' },
  }));
}
