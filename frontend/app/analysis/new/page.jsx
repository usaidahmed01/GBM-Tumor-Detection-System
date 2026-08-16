import { Suspense } from 'react';

import AnalysisIntakeWorkspace from '@/components/intake/AnalysisIntakeWorkspace';

function AnalysisIntakeFallback() {
  return (
    <main className="intake-shell intake-shell--loading" aria-busy="true">
      <section className="intake-stage-card glass-panel">
        <span className="eyebrow">NEUROGLIOMA AI · NEW MRI ANALYSIS</span>
        <h1>Preparing clinical intake…</h1>
        
      </section>
    </main>
  );
}

export default function NewAnalysisPage() {
  return (
    <Suspense fallback={<AnalysisIntakeFallback />}>
      <AnalysisIntakeWorkspace />
    </Suspense>
  );
}
