'use client';

import dynamic from 'next/dynamic';
import Link from 'next/link';
import { useEffect, useMemo, useState } from 'react';
import { AnimatePresence, motion, MotionConfig, useReducedMotion } from 'motion/react';

import { fetchViewerManifest, VIEWER_UI_VERSION } from '@/lib/api';
import { MRI_SEQUENCE_OPTIONS, SEGMENT_LEGEND } from '@/lib/viewerAssets';

const CornerstoneMprViewer = dynamic(() => import('./CornerstoneMprViewer'), {
  ssr: false,
  loading: () => <ViewerLoadingState label="Loading medical imaging engine…" />,
});

function Icon({ name, size = 18 }) {
  const common = {
    width: size,
    height: size,
    viewBox: '0 0 24 24',
    fill: 'none',
    stroke: 'currentColor',
    strokeWidth: 1.8,
    strokeLinecap: 'round',
    strokeLinejoin: 'round',
    'aria-hidden': true,
  };
  const paths = {
    home: <><path d="M3 11.5 12 4l9 7.5"/><path d="M5 10.5V20h14v-9.5"/><path d="M9.5 20v-6h5v6"/></>,
    layers: <><path d="m12 3-9 5 9 5 9-5-9-5Z"/><path d="m3 12 9 5 9-5"/><path d="m3 16 9 5 9-5"/></>,
    shield: <><path d="M12 3 5 6v5c0 4.5 2.9 8.2 7 10 4.1-1.8 7-5.5 7-10V6l-7-3Z"/><path d="m9 12 2 2 4-4"/></>,
    scan: <><path d="M7 3H5a2 2 0 0 0-2 2v2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/><path d="M17 21h2a2 2 0 0 0 2-2v-2"/><circle cx="12" cy="12" r="4"/><path d="M8 12h8M12 8v8"/></>,
    alert: <><path d="M12 3 2.7 20h18.6L12 3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/></>,
    check: <path d="m5 12 4 4L19 6"/>,
  };
  return <svg {...common}>{paths[name] || paths.scan}</svg>;
}

function ViewerLoadingState({ label = 'Loading study…' }) {
  return (
    <div className="viewer-loading glass-panel" role="status" aria-live="polite">
      <div className="viewer-loading__scanner">
        <div className="viewer-loading__brain" />
        <div className="viewer-loading__sweep" />
      </div>
      <div>
        <strong>{label}</strong>
        <span>Validating protected imaging assets and viewer state</span>
      </div>
    </div>
  );
}

function MeasurementCard({ item }) {
  const legend = SEGMENT_LEGEND.find((entry) => entry.region === item.region);
  return (
    <div className="metric-card">
      <div className="metric-card__top">
        <span className="metric-swatch" style={{ background: legend?.color }} />
        <span>{legend?.label || item.region}</span>
        <span className="metric-code">{item.region}</span>
      </div>
      <strong>{item.volume_cm3.toFixed(2)} <small>cm³</small></strong>
      <div className="metric-card__meta">
        <span>Max axial {item.max_axial_area_mm2.toFixed(1)} mm²</span>
        <span>{item.voxel_count.toLocaleString()} voxels</span>
      </div>
    </div>
  );
}

function RightSidebar({ manifest }) {
  const measurements = manifest?.quantification?.measurements || [];
  const localization = manifest?.localization || {};

  return (
    <aside className="clinical-sidebar">
      <section className="sidebar-section">
        <div className="section-heading">
          <div>
            <span className="eyebrow">QUANTIFICATION</span>
            <h2>Tumor burden</h2>
          </div>
          <span className={`mini-status ${manifest.quantification.available ? 'mini-status--ok' : ''}`}>
            {manifest.quantification.available ? 'Current' : manifest.quantification.stale ? 'Stale' : 'Unavailable'}
          </span>
        </div>
        {measurements.length ? (
          <div className="metric-stack">
            {measurements.map((item) => <MeasurementCard key={item.region} item={item} />)}
          </div>
        ) : (
          <p className="empty-copy">Physical measurements are not available for the current segmentation state.</p>
        )}
      </section>

      <section className="sidebar-section">
        <div className="section-heading">
          <div>
            <span className="eyebrow">ATLAS LOCALIZATION</span>
            <h2>Anatomical context</h2>
          </div>
          <span className={`mini-status ${localization.available ? 'mini-status--ok' : ''}`}>
            {localization.available ? 'Registered' : localization.stale ? 'Stale' : 'Unavailable'}
          </span>
        </div>
        {localization.available ? (
          <div className="localization-grid">
            <div><span>Hemisphere</span><strong>{localization.hemisphere || '—'}</strong></div>
            <div><span>Primary region</span><strong>{localization.primary_region || '—'}</strong></div>
            <div className="localization-grid__wide">
              <span>MNI centroid</span>
              <strong>{localization.centroid_mni_mm?.length === 3 ? localization.centroid_mni_mm.map((v) => Number(v).toFixed(1)).join(', ') + ' mm' : '—'}</strong>
            </div>
            <div className="localization-grid__wide">
              <span>Registration QC</span>
              <strong className={localization.registration_qc_passed ? 'text-good' : 'text-warn'}>
                {localization.registration_qc_passed ? 'Passed engineering gate' : 'Requires review'}
              </strong>
            </div>
          </div>
        ) : (
          <p className="empty-copy">Atlas-derived location is not current for this segmentation.</p>
        )}
      </section>

      <section className="sidebar-section sidebar-section--review">
        <div className="section-heading">
          <div>
            <span className="eyebrow">HUMAN REVIEW</span>
            <h2>Segmentation state</h2>
          </div>
          <Icon name="shield" />
        </div>
        <div className="review-state-row">
          <span className={`review-pill review-pill--${manifest.segmentation_review_status}`}>
            {manifest.segmentation_review_status}
          </span>
          {manifest.clinician_modified ? <span className="review-modified">Clinician modified</span> : null}
        </div>
        <p>
          Review, accept/reject and brush correction remain intentionally disabled in this step. They are implemented next with auditability and downstream recalculation.
        </p>
        <div className="review-actions review-actions--disabled" aria-label="Review controls coming in Phase 8 Step 3">
          <button disabled><Icon name="check" size={15} /> Accept</button>
          <button disabled>Correct mask</button>
          <button disabled>Reject</button>
        </div>
      </section>

      <section className="clinical-notice">
        <Icon name="alert" />
        <p>
          <strong>AI-assisted imaging review only.</strong> The SegResNet output delineates glioma-like regions; it is not a definitive GBM diagnosis. Clinician verification is required.
        </p>
      </section>
    </aside>
  );
}

export default function ViewerWorkspace({ studyUuid }) {
  const reduceMotion = useReducedMotion();
  const [manifest, setManifest] = useState(null);
  const [status, setStatus] = useState('loading');
  const [error, setError] = useState('');
  const [sequence, setSequence] = useState('T1C');
  const [overlayVisible, setOverlayVisible] = useState(true);
  const [overlayOpacity, setOverlayOpacity] = useState(0.42);
  const [activeTool, setActiveTool] = useState('window');
  const [resetToken, setResetToken] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    setStatus('loading');
    fetchViewerManifest(studyUuid, { signal: controller.signal })
      .then((payload) => {
        setManifest(payload);
        setStatus('ready');
      })
      .catch((reason) => {
        if (reason?.name === 'AbortError') return;
        setError(reason?.message || 'Unable to load the clinical viewer manifest.');
        setStatus('error');
      });
    return () => controller.abort();
  }, [studyUuid]);

  const shortStudyId = useMemo(() => `${studyUuid.slice(0, 8)}…${studyUuid.slice(-4)}`, [studyUuid]);

  if (status === 'loading') {
    return <main className="workspace-shell workspace-shell--center"><ViewerLoadingState /></main>;
  }

  if (status === 'error') {
    return (
      <main className="workspace-shell workspace-shell--center">
        <div className="error-panel glass-panel">
          <Icon name="alert" size={26} />
          <div><span className="eyebrow">VIEWER UNAVAILABLE</span><h1>Study could not be opened</h1><p>{error}</p></div>
          <Link href="/" className="button-link">Choose another study</Link>
        </div>
      </main>
    );
  }

  return (
    <MotionConfig reducedMotion="user">
      <main className="workspace-shell">
        <header className="workspace-topbar">
          <div className="topbar-brand">
            <Link href="/" className="icon-button" aria-label="Back to study launcher"><Icon name="home" /></Link>
            <div className="topbar-brand__mark"><Icon name="scan" /></div>
            <div>
              <span className="eyebrow">GBM CDSS · CLINICAL VIEWER</span>
              <div className="topbar-title-row">
                <h1>Multimodal MRI Review</h1>
                <span className="live-chip"><span /> Protected session</span>
              </div>
            </div>
          </div>
          <div className="study-meta">
            <div><span>Study</span><strong title={studyUuid}>{shortStudyId}</strong></div>
            <div><span>Source</span><strong>{manifest.source_format.toUpperCase()}</strong></div>
            <div><span>Orientation</span><strong>{manifest.canonical_orientation}</strong></div>
            <div><span>UI</span><strong>{VIEWER_UI_VERSION.replace('phase8_step2_', '').replace('_v1', '')}</strong></div>
          </div>
        </header>

        <motion.section
          className="workspace-main"
          initial={reduceMotion ? { opacity: 0 } : { opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35 }}
        >
          <div className="viewer-column">
            <div className="viewer-control-deck glass-panel">
              <div className="sequence-tabs" role="tablist" aria-label="MRI sequence">
                {MRI_SEQUENCE_OPTIONS.map((option) => (
                  <motion.button
                    key={option.value}
                    className={sequence === option.value ? 'sequence-tab sequence-tab--active' : 'sequence-tab'}
                    onClick={() => setSequence(option.value)}
                    whileTap={reduceMotion ? undefined : { scale: 0.97 }}
                    role="tab"
                    aria-selected={sequence === option.value}
                  >
                    <strong>{option.label}</strong><span>{option.description}</span>
                  </motion.button>
                ))}
              </div>

              <div className="overlay-control">
                <button
                  className={overlayVisible ? 'overlay-toggle overlay-toggle--on' : 'overlay-toggle'}
                  onClick={() => setOverlayVisible((value) => !value)}
                  aria-pressed={overlayVisible}
                >
                  <Icon name="layers" size={16} />
                  AI overlay
                  <span className="toggle-track"><span /></span>
                </button>
                <label>
                  <span>Opacity</span>
                  <input
                    type="range"
                    min="0.1"
                    max="0.8"
                    step="0.05"
                    value={overlayOpacity}
                    onChange={(event) => setOverlayOpacity(Number(event.target.value))}
                    disabled={!overlayVisible}
                  />
                  <output>{Math.round(overlayOpacity * 100)}%</output>
                </label>
              </div>
            </div>

            <CornerstoneMprViewer
              manifest={manifest}
              sequence={sequence}
              overlayVisible={overlayVisible}
              overlayOpacity={overlayOpacity}
              activeTool={activeTool}
              onActiveToolChange={setActiveTool}
              resetToken={resetToken}
              onReset={() => setResetToken((value) => value + 1)}
            />

            <div className="viewer-footer-strip">
              <div className="segmentation-legend">
                {SEGMENT_LEGEND.map((item) => (
                  <span key={item.region}><i style={{ background: item.color }} />{item.region}<small>{item.label}</small></span>
                ))}
              </div>
              <div className="viewer-provenance">
                <span>Segmentation {manifest.segmentation_uuid.slice(0, 8)}…</span>
                <span>Canonical {manifest.canonical_orientation}</span>
                <span>Checksums verified server-side</span>
              </div>
            </div>
          </div>

          <RightSidebar manifest={manifest} />
        </motion.section>

        <AnimatePresence>
          {manifest.quantification.stale || manifest.localization.stale ? (
            <motion.div
              className="stale-toast"
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 12 }}
            >
              <Icon name="alert" size={16} /> Some downstream measurements are stale and are not presented as current.
            </motion.div>
          ) : null}
        </AnimatePresence>
      </main>
    </MotionConfig>
  );
}
