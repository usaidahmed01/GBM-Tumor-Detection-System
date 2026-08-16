'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import ViewerWorkspace from '@/components/viewer/ViewerWorkspace';
import { getActiveCase } from '@/lib/activeCase';

export default function CurrentViewerPage() {
  const [activeCase, setActiveCaseState] = useState(undefined);
  useEffect(() => setActiveCaseState(getActiveCase()), []);

  if (activeCase === undefined) {
    return <main className="workspace-shell workspace-shell--center"><div className="viewer-loading glass-panel">Loading active NeuroGlioma AI case…</div></main>;
  }
  if (!activeCase?.studyUuid) {
    return <main className="workspace-shell workspace-shell--center"><div className="error-panel glass-panel"><div><span className="eyebrow">NO ACTIVE CASE</span><h1>Start or resume an analysis first</h1></div><Link href="/" className="button-link">Return home</Link></div></main>;
  }
  return <ViewerWorkspace studyUuid={activeCase.studyUuid} caseReference={activeCase.caseReference} />;
}
