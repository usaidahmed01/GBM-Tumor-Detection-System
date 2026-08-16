const ACTIVE_KEY = 'neuroglioma-ai-active-case-v1';
const RECENT_KEY = 'neuroglioma-ai-recent-cases-v1';
const MAX_RECENT = 6;

function browserReady() {
  return typeof window !== 'undefined';
}

export function getActiveCase() {
  if (!browserReady()) return null;
  try {
    return JSON.parse(window.sessionStorage.getItem(ACTIVE_KEY) || 'null');
  } catch {
    return null;
  }
}

export function setActiveCase(value) {
  if (!browserReady()) return;
  window.sessionStorage.setItem(ACTIVE_KEY, JSON.stringify(value));
  rememberCase(value);
}

export function patchActiveCase(patch) {
  const current = getActiveCase() || {};
  const next = { ...current, ...patch };
  setActiveCase(next);
  return next;
}

export function clearActiveCase() {
  if (!browserReady()) return;
  window.sessionStorage.removeItem(ACTIVE_KEY);
}

export function getRecentCases() {
  if (!browserReady()) return [];
  try {
    const value = JSON.parse(window.localStorage.getItem(RECENT_KEY) || '[]');
    return Array.isArray(value) ? value : [];
  } catch {
    return [];
  }
}

export function rememberCase(value) {
  if (!browserReady() || !value?.studyUuid || !value?.caseReference) return;
  const safe = {
    studyUuid: value.studyUuid,
    caseReference: value.caseReference,
    stage: value.stage || 'created',
    sourceFormat: value.sourceFormat || null,
    updatedAt: new Date().toISOString(),
  };
  const current = getRecentCases().filter((item) => item.studyUuid !== safe.studyUuid);
  window.localStorage.setItem(RECENT_KEY, JSON.stringify([safe, ...current].slice(0, MAX_RECENT)));
}
