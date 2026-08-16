export default function GlobalLoading() {
  return (
    <main className="route-loading-shell" role="status" aria-live="polite" aria-busy="true">
      <div className="route-loading-card">
        <div className="route-loading-mark"><span/></div>
        <strong>Opening NeuroGlioma AI</strong>
        <p>Preparing the next workspace…</p>
      </div>
    </main>
  );
}
