'use client';

import { use } from 'react';

import ReportWorkspace from '@/components/report/ReportWorkspace';

export default function ReportByStudyPage({ params }) {
  const { studyUuid } = use(params);
  return <ReportWorkspace studyUuid={studyUuid}/>;
}
