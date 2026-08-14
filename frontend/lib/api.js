export const VIEWER_UI_VERSION = 'phase8_step2_cornerstone3d_readonly_ui_v1';

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

export async function fetchViewerManifest(studyUuid, { signal } = {}) {
  const response = await fetch(
    `/gbm-api/studies/${encodeURIComponent(studyUuid)}/viewer/manifest`,
    {
      method: 'GET',
      cache: 'no-store',
      signal,
      headers: { Accept: 'application/json' },
    },
  );

  if (!response.ok) {
    throw new Error(await readApiError(response));
  }

  return response.json();
}
