'use client';

import Link from 'next/link';
import { useCallback, useEffect, useState } from 'react';
import { AnimatePresence, motion, MotionConfig, useReducedMotion } from 'motion/react';

import {
  fetchReviewHistory,
  fetchViewerManifest,
  runAnatomicalLocalization,
  runTumorQuantification,
  submitLabelmapCorrection,
  submitSegmentationReview,
} from '@/lib/api';
import { MRI_SEQUENCE_OPTIONS, SEGMENT_LEGEND } from '@/lib/viewerAssets';
import CornerstoneMprViewer from './CornerstoneMprViewer';

const SEQUENCE_SHORTCUTS = { T1C: '1', T1: '2', T2: '3', FLAIR: '4' };

function Icon({ name, size = 18 }) {
  const common = { width: size, height: size, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 1.8, strokeLinecap: 'round', strokeLinejoin: 'round', 'aria-hidden': true };
  const paths = {
    home: <><path d="M3 11.5 12 4l9 7.5"/><path d="M5 10.5V20h14v-9.5"/><path d="M9.5 20v-6h5v6"/></>,
    layers: <><path d="m12 3-9 5 9 5 9-5-9-5Z"/><path d="m3 12 9 5 9-5"/><path d="m3 16 9 5 9-5"/></>,
    shield: <><path d="M12 3 5 6v5c0 4.5 2.9 8.2 7 10 4.1-1.8 7-5.5 7-10V6l-7-3Z"/><path d="m9 12 2 2 4-4"/></>,
    scan: <><path d="M7 3H5a2 2 0 0 0-2 2v2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/><path d="M17 21h2a2 2 0 0 0 2-2v-2"/><circle cx="12" cy="12" r="4"/><path d="M8 12h8M12 8v8"/></>,
    alert: <><path d="M12 3 2.7 20h18.6L12 3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/></>,
    check: <path d="m5 12 4 4L19 6"/>,
    brush: <><path d="m14.5 4.5 5 5L10 19H5v-5l9.5-9.5Z"/><path d="m12 7 5 5"/></>,
    erase: <><path d="m7 18-3-3 9-9a2 2 0 0 1 3 0l2 2a2 2 0 0 1 0 3l-7 7H7Z"/><path d="M10 18h10"/></>,
    history: <><path d="M3 12a9 9 0 1 0 3-6.7"/><path d="M3 4v6h6"/><path d="M12 7v5l3 2"/></>,
    cube: <><path d="m12 3 8 4.5v9L12 21l-8-4.5v-9L12 3Z"/><path d="m4 7.5 8 4.5 8-4.5M12 12v9"/></>,
    x: <><path d="m6 6 12 12M18 6 6 18"/></>,
    file: <><path d="M6 2h8l4 4v16H6z"/><path d="M14 2v5h5"/><path d="M9 13h6M9 17h6"/></>,
  };
  return <svg {...common}>{paths[name] || paths.scan}</svg>;
}

function ViewerLoadingState({ label = 'Loading study…' }) {
  return <div className="viewer-loading glass-panel" role="status" aria-live="polite"><div className="viewer-loading__scanner"><div className="viewer-loading__brain"/><div className="viewer-loading__sweep"/></div><div><strong>{label}</strong><span>Loading imaging data</span></div></div>;
}

function MeasurementCard({ item }) {
  const legend = SEGMENT_LEGEND.find((entry) => entry.region === item.region);
  return <div className="metric-card"><div className="metric-card__top"><span className="metric-swatch" style={{ background: legend?.color }}/><span>{legend?.label || item.region}</span><span className="metric-code">{item.region}</span></div><strong>{item.volume_cm3.toFixed(2)} <small>cm³</small></strong><div className="metric-card__meta"><span>Max axial {item.max_axial_area_mm2.toFixed(1)} mm²</span><span>{item.voxel_count.toLocaleString()} voxels</span></div></div>;
}

function ReviewPanel({
  manifest,
  correctionOpen,
  onOpenCorrection,
  onCancelCorrection,
  onReview,
  reviewBusy,
  note,
  setNote,
  history,
}) {
  return (
    <section className="sidebar-section sidebar-section--review">
      <div className="section-heading">
        <div><span className="eyebrow">HUMAN REVIEW</span><h2>Segmentation state</h2></div>
        <Icon name="shield" />
      </div>
      <div className="review-state-row">
        <span className={`review-pill review-pill--${manifest.segmentation_review_status}`}>{manifest.segmentation_review_status}</span>
        {manifest.clinician_modified ? <span className="review-modified">Clinician modified</span> : null}
      </div>
      <p>Review the segmentation in all three planes before finalizing.</p>
      <textarea
        className="review-note"
        maxLength={1000}
        value={note}
        onChange={(event) => setNote(event.target.value)}
        placeholder="Optional clinician review note…"
        aria-label="Clinician review note"
      />
      <div className="review-actions">
        <motion.button whileTap={{ scale: 0.97 }} disabled={reviewBusy || correctionOpen} aria-busy={reviewBusy} className={reviewBusy ? "review-action review-action--accept is-working" : "review-action review-action--accept"} onClick={() => onReview('accept')}>{reviewBusy ? <span className="button-working-mark" aria-hidden="true"/> : <Icon name="check" size={15}/>}Accept</motion.button>
        <motion.button whileTap={{ scale: 0.97 }} disabled={reviewBusy} className="review-action review-action--correct" onClick={correctionOpen ? onCancelCorrection : onOpenCorrection}><Icon name={correctionOpen ? 'x' : 'brush'} size={15}/>{correctionOpen ? 'Cancel edit' : 'Correct mask'}</motion.button>
        <motion.button whileTap={{ scale: 0.97 }} disabled={reviewBusy || correctionOpen} aria-busy={reviewBusy} className={reviewBusy ? "review-action review-action--reject is-working" : "review-action review-action--reject"} onClick={() => onReview('reject')}>{reviewBusy ? <span className="button-working-mark" aria-hidden="true"/> : <Icon name="x" size={15}/>}Reject</motion.button>
      </div>
      <div className="review-history-mini"><Icon name="history" size={14}/><span>{history?.revisions?.length || 0} review revision{history?.revisions?.length === 1 ? '' : 's'}</span></div>
    </section>
  );
}

function RightSidebar(props) {
  const { manifest, onRetryQuantification, onRetryLocalization, derivedBusy } = props;
  const measurements = manifest?.quantification?.measurements || [];
  const localization = manifest?.localization || {};
  return (
    <aside className="clinical-sidebar">
      <section className="sidebar-section">
        <div className="section-heading"><div><span className="eyebrow">QUANTIFICATION</span><h2>Tumor burden</h2></div><span className={`mini-status ${manifest.quantification.available ? 'mini-status--ok' : ''}`}>{manifest.quantification.available ? 'Current' : manifest.quantification.stale ? 'Stale' : 'Unavailable'}</span></div>
        {measurements.length ? <div className="metric-stack">{measurements.map((item) => <MeasurementCard key={item.region} item={item}/>)}</div> : <div className="sidebar-empty-action"><p className="empty-copy">Physical measurements are not available for the current segmentation state.</p>{manifest.segmentation_review_status !== 'rejected' ? <button type="button" className={derivedBusy === 'quantification' ? 'sidebar-inline-action is-working' : 'sidebar-inline-action'} disabled={Boolean(derivedBusy)} onClick={onRetryQuantification}>{derivedBusy === 'quantification' ? <span className="button-working-mark" aria-hidden="true"/> : null}{derivedBusy === 'quantification' ? 'Recalculating…' : 'Recalculate'}</button> : null}</div>}
      </section>
      <section className="sidebar-section">
        <div className="section-heading"><div><span className="eyebrow">ATLAS LOCALIZATION</span><h2>Anatomical context</h2></div><span className={`mini-status ${localization.available ? 'mini-status--ok' : ''}`}>{localization.available ? 'Registered' : localization.stale ? 'Stale' : 'Unavailable'}</span></div>
        {localization.available ? <div className="localization-grid"><div><span>Hemisphere</span><strong>{localization.hemisphere || '—'}</strong></div><div><span>Primary region</span><strong>{localization.primary_region || '—'}</strong></div><div className="localization-grid__wide"><span>MNI centroid</span><strong>{localization.centroid_mni_mm?.length === 3 ? `${localization.centroid_mni_mm.map((v) => Number(v).toFixed(1)).join(', ')} mm` : '—'}</strong></div><div className="localization-grid__wide"><span>Registration QC</span><strong className={localization.registration_qc_passed ? 'text-good' : 'text-warn'}>{localization.registration_qc_passed ? 'Validated' : 'Requires review'}</strong></div></div> : <div className="sidebar-empty-action"><p className="empty-copy">Atlas-derived location is not current for this segmentation.</p>{manifest.segmentation_review_status !== 'rejected' ? <button type="button" className={derivedBusy === 'localization' ? 'sidebar-inline-action is-working' : 'sidebar-inline-action'} disabled={Boolean(derivedBusy)} onClick={onRetryLocalization}>{derivedBusy === 'localization' ? <span className="button-working-mark" aria-hidden="true"/> : null}{derivedBusy === 'localization' ? 'Retrying…' : 'Retry localization'}</button> : null}</div>}
      </section>
      <ReviewPanel {...props}/>
      <section className="clinical-notice"><Icon name="alert"/><p><strong>AI-assisted imaging review only.</strong> The SegResNet output delineates glioma-like regions; it is not a definitive GBM diagnosis. Clinician verification is required.</p></section>
    </aside>
  );
}

function CorrectionDeck({ mode, setMode, segmentIndex, setSegmentIndex, brushSize, setBrushSize, dirty, busy, onSave, onCancel }) {
  return (
    <motion.div className="correction-deck" initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }}>
      <div className="correction-deck__title"><span className="correction-beacon"><i/></span><div><span className="eyebrow">CLINICIAN CORRECTION MODE</span><strong>Edit the segmentation in any MPR plane</strong></div></div>
      <div className="correction-toolset">
        <div className="correction-mode-tabs"><button className={mode === 'paint' ? 'active' : ''} onClick={() => setMode('paint')}><Icon name="brush" size={15}/>Paint</button><button className={mode === 'erase' ? 'active' : ''} onClick={() => setMode('erase')}><Icon name="erase" size={15}/>Erase</button></div>
        <div className="segment-picker" aria-label="Active segmentation region">{SEGMENT_LEGEND.map((segment) => <button key={segment.index} onClick={() => setSegmentIndex(segment.index)} className={segmentIndex === segment.index ? 'segment-choice active' : 'segment-choice'}><i style={{ background: segment.color }}/><strong>{segment.region}</strong><span>{segment.label}</span></button>)}</div>
        <label className="brush-size"><span>Brush</span><input type="range" min="3" max="40" step="1" value={brushSize} onChange={(event) => setBrushSize(Number(event.target.value))}/><output>{brushSize}px</output></label>
      </div>
      <div className="correction-deck__actions"><span>{dirty ? 'Unsaved changes' : 'Paint or erase with the left mouse button'}</span><button className="ghost-button" onClick={onCancel} disabled={busy}>Cancel</button><button className={busy ? "save-correction is-working" : "save-correction"} onClick={onSave} disabled={busy || !dirty} aria-busy={busy}>{busy ? <><span className="button-working-mark" aria-hidden="true"/>Saving changes…</> : 'Save correction & recalculate'}</button></div>
    </motion.div>
  );
}

export default function ViewerWorkspace({ studyUuid, caseReference = null }) {
  const reduceMotion = useReducedMotion();
  const [manifest, setManifest] = useState(null);
  const [history, setHistory] = useState(null);
  const [status, setStatus] = useState('loading');
  const [error, setError] = useState('');
  const [sequence, setSequence] = useState('T1C');
  const [overlayVisible, setOverlayVisible] = useState(true);
  const [overlayOpacity, setOverlayOpacity] = useState(0.42);
  const [activeTool, setActiveTool] = useState('window');
  const [threeDVisible, setThreeDVisible] = useState(false);
  const [threeDMode, setThreeDMode] = useState('composite');
  const [resetToken, setResetToken] = useState(0);
  const [viewerReloadToken, setViewerReloadToken] = useState(0);
  const [correctionOpen, setCorrectionOpen] = useState(false);
  const [editMode, setEditMode] = useState('paint');
  const [editSegmentIndex, setEditSegmentIndex] = useState(2);
  const [brushSize, setBrushSize] = useState(12);
  const [editorController, setEditorController] = useState(null);
  const [dirty, setDirty] = useState(false);
  const [note, setNote] = useState('');
  const [reviewBusy, setReviewBusy] = useState(false);
  const [derivedBusy, setDerivedBusy] = useState(null);
  const [toast, setToast] = useState(null);

  const reloadViewerState = useCallback(async () => {
    const [nextManifest, nextHistory] = await Promise.all([
      fetchViewerManifest(studyUuid),
      fetchReviewHistory(studyUuid).catch(() => null),
    ]);
    setManifest(nextManifest);
    setHistory(nextHistory);
    return nextManifest;
  }, [studyUuid]);

  useEffect(() => {
    const controller = new AbortController();
    let cancelled = false;

    async function loadViewer() {
      setStatus('loading'); setError('');
      try {
        let payload = await fetchViewerManifest(studyUuid, { signal: controller.signal });
        const needsDerivedRecovery = payload?.segmentation_review_status !== 'rejected'
          && (!payload?.quantification?.available || !payload?.localization?.available);

        if (needsDerivedRecovery && !cancelled) {
          setStatus('deriving');
          try { await runTumorQuantification(studyUuid); } catch {}
          try { await runAnatomicalLocalization(studyUuid); } catch {}
          payload = await fetchViewerManifest(studyUuid, { signal: controller.signal });
        }

        const reviewHistory = await fetchReviewHistory(studyUuid).catch(() => null);
        if (cancelled) return;
        setManifest(payload); setHistory(reviewHistory); setStatus('ready');
      } catch (reason) {
        if (reason?.name === 'AbortError' || cancelled) return;
        setError(reason?.message || 'Unable to load the clinical viewer.'); setStatus('error');
      }
    }

    loadViewer();
    return () => { cancelled = true; controller.abort(); };
  }, [studyUuid]);

  const showToast = (kind, message) => {
    setToast({ kind, message, id: Date.now() });
    window.setTimeout(() => setToast(null), 5200);
  };

  useEffect(() => {
    const onKeyDown = (event) => {
      const target = event.target;
      if (target?.matches?.('input, textarea, select, [contenteditable="true"]')) return;
      if (correctionOpen) return;
      const key = event.key.toLowerCase();
      const sequenceByKey = { '1': 'T1C', '2': 'T1', '3': 'T2', '4': 'FLAIR' };
      if (sequenceByKey[key]) {
        event.preventDefault();
        setSequence(sequenceByKey[key]);
      } else if (key === 'o') {
        event.preventDefault();
        setOverlayVisible((value) => !value);
      } else if (key === 'v') {
        event.preventDefault();
        setThreeDVisible((value) => !value);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [correctionOpen]);

  const retryDerivedOutput = async (kind) => {
    setDerivedBusy(kind);
    try {
      if (kind === 'quantification') await runTumorQuantification(studyUuid);
      else await runAnatomicalLocalization(studyUuid);
      await reloadViewerState();
      showToast('success', kind === 'quantification' ? 'Tumor measurements updated.' : 'Anatomical localization updated.');
    } catch (reason) {
      showToast('error', reason?.message || (kind === 'quantification' ? 'Tumor measurements could not be recalculated.' : 'Anatomical localization could not be updated.'));
    } finally {
      setDerivedBusy(null);
    }
  };

  const handleReview = async (action) => {
    setReviewBusy(true);
    try {
      await submitSegmentationReview(studyUuid, action, note);
      await reloadViewerState();
      setNote('');
      showToast(action === 'accept' ? 'success' : 'warning', action === 'accept' ? 'Segmentation accepted. Review status updated.' : 'Segmentation rejected. Volume and atlas location are now blocked for this mask.');
    } catch (reason) {
      showToast('error', reason?.message || 'Review action failed.');
    } finally { setReviewBusy(false); }
  };

  const openCorrection = () => { setCorrectionOpen(true); setEditMode('paint'); setDirty(false); setOverlayVisible(true); };
  const cancelCorrection = () => { setCorrectionOpen(false); setDirty(false); setEditMode('paint'); setViewerReloadToken((value) => value + 1); };

  const saveCorrection = async () => {
    if (!editorController?.exportRawLabelmap || !manifest) return;
    setReviewBusy(true);
    try {
      const rawLabelmap = editorController.exportRawLabelmap();
      const labelmapAsset = manifest.assets.find((asset) => asset.alias === 'mask_labelmap');
      const result = await submitLabelmapCorrection(studyUuid, {
        rawLabelmap,
        sourceChecksumSha256: labelmapAsset.checksum_sha256,
        note,
      });
      await reloadViewerState();
      setCorrectionOpen(false); setDirty(false); setNote('');
      const loc = result?.downstream?.localization;
      showToast(loc === 'recalculated' ? 'success' : 'warning', loc === 'recalculated' ? 'Correction saved. Volume and anatomical location were recalculated.' : 'Correction saved and volume recalculated. Localization requires review/retry; stale location is hidden.');
    } catch (reason) {
      showToast('error', reason?.message || 'Mask correction could not be saved.');
    } finally { setReviewBusy(false); }
  };

  if (status === 'loading' || status === 'deriving') return <main className="workspace-shell workspace-shell--center"><ViewerLoadingState label={status === 'deriving' ? 'Preparing measurements…' : 'Loading MRI viewer…'}/></main>;
  if (status === 'error') return <main className="workspace-shell workspace-shell--center"><div className="error-panel glass-panel"><Icon name="alert" size={26}/><div><span className="eyebrow">VIEWER UNAVAILABLE</span><h1>Study could not be opened</h1><p>{error}</p></div><div className="error-panel__actions"><button type="button" className="button-link" onClick={() => window.location.reload()}>Retry</button><Link href="/" className="button-link button-link--secondary">Return home</Link></div></div></main>;

  return (
    <MotionConfig reducedMotion="user">
      <main className="workspace-shell">
        <header className="workspace-topbar">
          <div className="topbar-brand"><Link href="/" className="icon-button" aria-label="Back to study launcher"><Icon name="home"/></Link><div className="topbar-brand__mark"><Icon name="scan"/></div><div><span className="eyebrow">NEUROGLIOMA AI · CLINICAL VIEWER</span><div className="topbar-title-row"><h1>Multimodal MRI Review</h1><span className="live-chip"><span/>MRI review</span></div></div></div>
          <div className="workspace-topbar__right"><div className="study-meta"><div><span>Case</span><strong>{caseReference || "Current analysis"}</strong></div><div><span>Source</span><strong>{manifest.source_format.toUpperCase()}</strong></div><div><span>Orientation</span><strong>{manifest.canonical_orientation}</strong></div><div><span>Review</span><strong>{manifest.segmentation_review_status.replaceAll('_',' ')}</strong></div></div><Link href="/report/current" className="viewer-report-button"><Icon name="file" size={15}/>Report</Link></div>
        </header>

        <motion.section className="workspace-main" initial={reduceMotion ? { opacity: 0 } : { opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35 }}>
          <div className="viewer-column">
            <div className="viewer-control-deck glass-panel">
              <div className="sequence-tabs" role="tablist" aria-label="MRI sequence">
                {MRI_SEQUENCE_OPTIONS.map((option) => (
                  <motion.button
                    key={option.value}
                    className={sequence === option.value ? 'sequence-tab sequence-tab--active' : 'sequence-tab'}
                    disabled={correctionOpen}
                    onClick={() => setSequence(option.value)}
                    whileTap={reduceMotion ? undefined : { scale: 0.97 }}
                    role="tab"
                    aria-selected={sequence === option.value}
                    aria-keyshortcuts={SEQUENCE_SHORTCUTS[option.value]}
                    title={`${option.label} · ${SEQUENCE_SHORTCUTS[option.value]}`}
                  >
                    <strong>{option.label}</strong><span>{option.description}</span><kbd>{SEQUENCE_SHORTCUTS[option.value]}</kbd>
                  </motion.button>
                ))}
              </div>
              <div className="overlay-control">
                <button
                  disabled={correctionOpen}
                  className={overlayVisible ? 'overlay-toggle overlay-toggle--on' : 'overlay-toggle'}
                  onClick={() => setOverlayVisible((value) => !value)}
                  aria-pressed={overlayVisible}
                  aria-keyshortcuts="O"
                  title="Toggle AI overlay · O"
                >
                  <Icon name="layers" size={16}/>AI overlay<kbd>O</kbd><span className="toggle-track"><span/></span>
                </button>
                <label><span>Opacity</span><input type="range" min="0.1" max="0.8" step="0.05" value={overlayOpacity} onChange={(event) => setOverlayOpacity(Number(event.target.value))} disabled={!overlayVisible}/><output>{Math.round(overlayOpacity * 100)}%</output></label>
              </div>
              <div className="three-d-launch-control">
                <button
                  disabled={correctionOpen}
                  className={threeDVisible ? 'three-d-toggle three-d-toggle--on' : 'three-d-toggle'}
                  onClick={() => setThreeDVisible((value) => !value)}
                  aria-pressed={threeDVisible}
                  aria-keyshortcuts="V"
                  title="Toggle 3D review · V"
                >
                  <Icon name="cube" size={16}/>
                  <span><strong>3D review</strong><small>{threeDVisible ? 'Volume viewport active' : 'Open on demand'}</small></span>
                  <kbd>V</kbd><i>{threeDVisible ? 'ON' : 'OFF'}</i>
                </button>
              </div>
            </div>

            <AnimatePresence>{correctionOpen ? <CorrectionDeck mode={editMode} setMode={setEditMode} segmentIndex={editSegmentIndex} setSegmentIndex={setEditSegmentIndex} brushSize={brushSize} setBrushSize={setBrushSize} dirty={dirty} busy={reviewBusy} onSave={saveCorrection} onCancel={cancelCorrection}/> : null}</AnimatePresence>

            <CornerstoneMprViewer
              manifest={manifest}
              sequence={sequence}
              overlayVisible={overlayVisible}
              overlayOpacity={overlayOpacity}
              activeTool={activeTool}
              onActiveToolChange={setActiveTool}
              resetToken={resetToken}
              onReset={() => setResetToken((value) => value + 1)}
              editMode={correctionOpen ? editMode : 'off'}
              editSegmentIndex={editSegmentIndex}
              brushSize={brushSize}
              onEditorController={setEditorController}
              onPotentialEdit={() => setDirty(true)}
              reloadToken={viewerReloadToken}
              threeDVisible={threeDVisible}
              threeDMode={threeDMode}
              onThreeDModeChange={setThreeDMode}
            />

            <div className="viewer-footer-strip"><div className="segmentation-legend">{SEGMENT_LEGEND.map((item) => <span key={item.region}><i style={{ background: item.color }}/>{item.region}<small>{item.label}</small></span>)}</div><div className="viewer-provenance"><span>Segmentation review</span><span>{manifest.canonical_orientation} orientation</span><span>{threeDVisible ? `3D ${threeDMode.toUpperCase()} review` : '3D review available'}</span><span className="viewer-shortcuts-summary">1–4 sequences · W/P/Z tools · O overlay · V 3D</span></div></div>
          </div>

          <RightSidebar manifest={manifest} correctionOpen={correctionOpen} onOpenCorrection={openCorrection} onCancelCorrection={cancelCorrection} onReview={handleReview} reviewBusy={reviewBusy} note={note} setNote={setNote} history={history} derivedBusy={derivedBusy} onRetryQuantification={() => retryDerivedOutput('quantification')} onRetryLocalization={() => retryDerivedOutput('localization')}/>
        </motion.section>

        <AnimatePresence>
          {manifest.quantification.stale || manifest.localization.stale ? <motion.div className="stale-toast" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 12 }}><Icon name="alert" size={16}/>Some downstream measurements are stale and are not presented as current.</motion.div> : null}
          {toast ? <motion.div key={toast.id} className={`review-toast review-toast--${toast.kind}`} initial={{ opacity: 0, y: 18, scale: .98 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: 12 }}><span className="review-toast__pulse"/><p>{toast.message}</p></motion.div> : null}
        </AnimatePresence>
      </main>
    </MotionConfig>
  );
}
