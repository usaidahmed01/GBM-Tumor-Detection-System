'use client';

import Link from 'next/link';
import { useEffect, useState } from 'react';

import ReportWorkspace from '@/components/report/ReportWorkspace';
import { getActiveCase } from '@/lib/activeCase';

export default function CurrentReportPage() {
  const [activeCase, setActiveCaseState] = useState(undefined);

  useEffect(() => {
    setActiveCaseState(getActiveCase());
  }, []);

  if (activeCase === undefined) {
    return <main className="report-shell report-shell--center"><div className="report-loader" aria-label="Loading report"/></main>;
  }

  if (!activeCase?.studyUuid) {
    return (
      <main className="report-shell report-shell--center">
        <section className="report-empty glass-panel">
          <span className="eyebrow">NEUROGLIOMA AI · REPORT</span>
          <h1>No active analysis</h1>
          <p>Open or complete an MRI analysis before viewing its structured report.</p>
          <Link href="/" className="report-primary-button">Return to dashboard</Link>
        </section>
      </main>
    );
  }

  return <ReportWorkspace studyUuid={activeCase.studyUuid} caseReference={activeCase.caseReference}/>; 
}
