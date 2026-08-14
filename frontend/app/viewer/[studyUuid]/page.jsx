import ViewerWorkspace from '@/components/viewer/ViewerWorkspace';

export default async function StudyViewerPage({ params }) {
  const { studyUuid } = await params;
  return <ViewerWorkspace studyUuid={studyUuid} />;
}
