'use client';

import Link from 'next/link';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { AnimatePresence, motion, MotionConfig, useReducedMotion } from 'motion/react';

import {
  fetchCurrentReport,
  fetchReportPreview,
  finalizeClinicalReport,
  fuseStudyDecision,
} from '@/lib/api';

const REPORT_UI_VERSION = 'phase9_step4_report_ui_v1';

function Icon({ name, size = 18 }) {
  const common = { width: size, height: size, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 1.8, strokeLinecap: 'round', strokeLinejoin: 'round', 'aria-hidden': true };
  const icons = {
    arrow: <><path d="M5 12h14"/><path d="m13 6 6 6-6 6"/></>,
    back: <><path d="m15 18-6-6 6-6"/></>,
    check: <path d="m5 12 4 4L19 6"/>,
    shield: <><path d="M12 3 5 6v5c0 4.5 2.9 8.2 7 10 4.1-1.8 7-5.5 7-10V6l-7-3Z"/><path d="m9 12 2 2 4-4"/></>,
    print: <><path d="M6 9V3h12v6"/><path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"/><path d="M6 14h12v7H6z"/></>,
    file: <><path d="M6 2h8l4 4v16H6z"/><path d="M14 2v5h5"/><path d="M9 13h6M9 17h6"/></>,
    alert: <><path d="M12 3 2.7 20h18.6L12 3Z"/><path d="M12 9v4M12 17h.01"/></>,
    lock: <><rect x="5" y="10" width="14" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3"/></>,
    download: <><path d="M12 3v12"/><path d="m7 10 5 5 5-5"/><path d="M5 21h14"/></>,
  };
  return <svg {...common}>{icons[name] || icons.file}</svg>;
}

function formatDate(value) {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(date);
}

function valueOrDash(value) {
  return value === null || value === undefined || value === '' ? '—' : String(value);
}

function decisionLabel(state) {
  const labels = {
    gbm_suspected: 'GBM suspected',
    gbm_not_suspected: 'GBM not suspected',
    indeterminate: 'Indeterminate / unable to assess',
    pending: 'Pending',
  };
  return labels[state] || valueOrDash(state);
}

function stateTone(state) {
  if (state === 'gbm_suspected') return 'warn';
  if (state === 'gbm_not_suspected') return 'good';
  return 'neutral';
}

function ReportField({ label, value, mono = false }) {
  return <div className="report-field"><span>{label}</span><strong className={mono ? 'mono' : ''}>{valueOrDash(value)}</strong></div>;
}

function ReportSection({ index, title, children, className = '' }) {
  return <section className={`report-section ${className}`}><header><span>{index}</span><h2>{title}</h2></header><div className="report-section__body">{children}</div></section>;
}

function ReportLoading() {
  return <main className="report-shell report-shell--center"><div className="report-loader"/><p className="report-loader-copy">Preparing structured report…</p></main>;
}

export default function ReportWorkspace({ studyUuid, caseReference = null }) {
  const reduceMotion = useReducedMotion();
  const [preview, setPreview] = useState(null);
  const [finalized, setFinalized] = useState(null);
  const [status, setStatus] = useState('loading');
  const [error, setError] = useState('');
  const [clinicianName, setClinicianName] = useState('');
  const [clinicianComment, setClinicianComment] = useState('');
  const [signing, setSigning] = useState(false);
  const [notice, setNotice] = useState('');

  const load = useCallback(async () => {
    setStatus('loading');
    setError('');
    try {
      const existing = await fetchCurrentReport(studyUuid).catch(() => null);
      if (existing) {
        setFinalized(existing);
        setPreview(null);
        setStatus('ready');
        return;
      }
      await fuseStudyDecision(studyUuid).catch(() => null);
      const nextPreview = await fetchReportPreview(studyUuid);
      setPreview(nextPreview);
      setFinalized(null);
      setStatus('ready');
    } catch (reason) {
      setError(reason?.message || 'Unable to prepare the structured report.');
      setStatus('error');
    }
  }, [studyUuid]);

  useEffect(() => { load(); }, [load]);

  const response = finalized || preview;
  const report = response?.report || null;
  const isFinal = Boolean(finalized);
  const blockers = preview?.blockers || [];
  const readyToFinalize = Boolean(preview?.finalization_ready);

  const patientStudy = report?.patient_study || {};
  const clinical = report?.clinical_context || {};
  const validation = report?.input_validation || {};
  const assessment = report?.gbm_assessment || {};
  const tumor = report?.tumor_analysis || null;
  const quant = tumor?.quantification || null;
  const localization = tumor?.localization || null;
  const review = report?.human_review || {};
  const trace = report?.traceability || {};

  const safeCaseReference = caseReference || patientStudy.case_reference || 'Current analysis';
  const probabilityText = useMemo(() => {
    const p = assessment.calibrated_probability_gbm;
    return typeof p === 'number' ? `${(p * 100).toFixed(1)}%` : 'Not available';
  }, [assessment.calibrated_probability_gbm]);

  const handleFinalize = async (event) => {
    event.preventDefault();
    if (!clinicianName.trim() || !readyToFinalize) return;
    setSigning(true);
    setError('');
    try {
      const result = await finalizeClinicalReport(studyUuid, {
        clinician_name: clinicianName.trim(),
        clinician_comment: clinicianComment.trim() || null,
      });
      setFinalized(result);
      setPreview(null);
      setNotice('Report finalized successfully.');
    } catch (reason) {
      setError(reason?.message || 'Report could not be finalized.');
    } finally {
      setSigning(false);
    }
  };

  const exportJson = () => {
    if (!response) return;
    const blob = new Blob([JSON.stringify(response, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `NeuroGliomaAI-${safeCaseReference}-report.json`.replace(/[^a-zA-Z0-9._-]/g, '_');
    anchor.click();
    URL.revokeObjectURL(url);
  };

  if (status === 'loading') return <ReportLoading/>;
  if (status === 'error' && !report) {
    return <main className="report-shell report-shell--center"><section className="report-empty glass-panel"><Icon name="alert" size={28}/><span className="eyebrow">REPORT UNAVAILABLE</span><h1>Report could not be prepared</h1><p>{error}</p><div className="report-empty__actions"><Link href="/viewer/current" className="report-secondary-button">Return to viewer</Link><button onClick={load} className="report-primary-button">Retry</button></div></section></main>;
  }

  return (
    <MotionConfig reducedMotion="user">
      <main className="report-shell">
        <header className="report-appbar no-print">
          <div className="report-appbar__brand"><Link href="/viewer/current" className="report-icon-button" aria-label="Back to clinical viewer"><Icon name="back"/></Link><div className="report-brand-mark"><Icon name="file"/></div><div><span className="eyebrow">NEUROGLIOMA AI · STRUCTURED REPORT</span><strong>{safeCaseReference}</strong></div></div>
          <div className="report-appbar__actions"><span className={isFinal ? 'report-status-chip report-status-chip--final' : 'report-status-chip'}><i/>{isFinal ? 'Finalized' : readyToFinalize ? 'Ready for sign-off' : 'Preview'}</span><button className="report-secondary-button" onClick={exportJson}><Icon name="download" size={15}/>Export JSON</button><button className="report-primary-button" onClick={() => window.print()}><Icon name="print" size={15}/>Print / Save PDF</button></div>
        </header>

        <motion.div className="report-layout" initial={reduceMotion ? {opacity:0}:{opacity:0,y:10}} animate={{opacity:1,y:0}} transition={{duration:.35}}>
          <article className="report-paper">
            <div className="report-paper__masthead"><div><span className="eyebrow">NEURO-ONCOLOGY MRI DECISION SUPPORT</span><h1>NeuroGlioma AI</h1><p>Structured imaging assessment report</p></div><div className="report-case-block"><span>Case reference</span><strong>{safeCaseReference}</strong><small>{isFinal ? `Signed ${formatDate(finalized.signed_at)}` : 'Report preview'}</small></div></div>

            <section className={`report-decision-banner report-decision-banner--${stateTone(assessment.state)}`}><div><span className="eyebrow">GBM ASSESSMENT</span><h2>{decisionLabel(assessment.state)}</h2><p>{assessment.summary || 'No additional summary available.'}</p></div><div className="report-probability"><span>Calibrated probability</span><strong>{probabilityText}</strong><small>{assessment.calibrated_probability_gbm == null ? 'Classifier evidence unavailable for this input path' : 'Interpret with safety gates and clinical context'}</small></div></section>

            <div className="report-columns">
              <ReportSection index="01" title="Patient & study">
                <div className="report-field-grid"><ReportField label="Case reference" value={patientStudy.case_reference}/><ReportField label="Patient name" value={patientStudy.patient_name}/><ReportField label="Age" value={patientStudy.age_years}/><ReportField label="Sex" value={patientStudy.sex}/><ReportField label="MRI date" value={patientStudy.mri_date}/><ReportField label="Assessment created" value={formatDate(patientStudy.assessment_created_at)}/></div>
              </ReportSection>

              <ReportSection index="02" title="Clinical context">
                <div className="report-field-grid"><ReportField label="Symptoms" value={(clinical.symptoms || []).join(', ') || 'None documented'}/><ReportField label="Symptom duration" value={clinical.symptom_duration}/><ReportField label="Prior treatment" value={clinical.prior_treatment ? 'Yes' : 'No'}/><ReportField label="Clinical notes" value={clinical.clinical_notes}/></div>
              </ReportSection>
            </div>

            <ReportSection index="03" title="Input validation">
              <div className="report-validation-grid"><ReportField label="Source format" value={validation.source_format?.toUpperCase()}/><ReportField label="Modality" value={validation.modality}/><ReportField label="MRI QC" value={validation.qc_status?.toUpperCase()}/><ReportField label="Brain scope" value={validation.brain_scope_status?.replaceAll('_',' ')}/><ReportField label="Routing" value={validation.capability_routing_status?.replaceAll('_',' ')}/></div>
            </ReportSection>

            <ReportSection index="04" title="Tumor analysis">
              {tumor ? <><div className="report-metric-grid"><div><span>WT</span><strong>{quant?.wt_volume_cm3 != null ? `${Number(quant.wt_volume_cm3).toFixed(2)} cm³` : '—'}</strong><small>{valueOrDash(tumor.wt_voxel_count)} voxels</small></div><div><span>TC</span><strong>{quant?.tc_volume_cm3 != null ? `${Number(quant.tc_volume_cm3).toFixed(2)} cm³` : '—'}</strong><small>{valueOrDash(tumor.tc_voxel_count)} voxels</small></div><div><span>ET</span><strong>{quant?.et_volume_cm3 != null ? `${Number(quant.et_volume_cm3).toFixed(2)} cm³` : '—'}</strong><small>{valueOrDash(tumor.et_voxel_count)} voxels</small></div></div><div className="report-field-grid report-field-grid--spaced"><ReportField label="Segmentation review" value={tumor.segmentation_review_status?.toUpperCase()}/><ReportField label="Clinician modified" value={tumor.clinician_modified ? 'Yes' : 'No'}/><ReportField label="Max axial WT area" value={quant?.wt_max_axial_area_mm2 != null ? `${Number(quant.wt_max_axial_area_mm2).toFixed(1)} mm²` : null}/><ReportField label="Hemisphere" value={localization?.hemisphere}/><ReportField label="Primary region" value={localization?.primary_region}/><ReportField label="MNI centroid" value={localization?.centroid_mni_mm?.length ? localization.centroid_mni_mm.map((v)=>Number(v).toFixed(1)).join(', ') + ' mm' : null}/></div></> : <p className="report-muted-copy">No current validated 3D segmentation is available for this report.</p>}
            </ReportSection>

            <div className="report-columns">
              <ReportSection index="05" title="Human review">
                <div className="report-field-grid"><ReportField label="Segmentation available" value={review.segmentation_available ? 'Yes' : 'No'}/><ReportField label="Review status" value={review.segmentation_review_status?.toUpperCase()}/><ReportField label="Clinician modified" value={review.clinician_modified ? 'Yes' : 'No'}/></div>
              </ReportSection>
              <ReportSection index="06" title="Traceability">
                <div className="report-trace"><ReportField label="Decision fusion" value={trace.decision_fusion_version} mono/><ReportField label="Decision fused" value={formatDate(trace.decision_fused_at)}/>{trace.classifier_model ? <ReportField label="Classifier" value={`${trace.classifier_model.name} · ${trace.classifier_model.version}`}/> : null}{trace.segmentation_model ? <ReportField label="Segmentation" value={`${trace.segmentation_model.name} · ${trace.segmentation_model.version}`}/> : null}</div>
              </ReportSection>
            </div>

            {assessment.safety_reason_codes?.length ? <ReportSection index="07" title="Safety conditions"><div className="report-code-list">{assessment.safety_reason_codes.map((code)=><span key={code}>{code.replaceAll('_',' ')}</span>)}</div></ReportSection> : null}

            <section className="report-notice"><Icon name="shield"/><div><strong>Clinical decision-support notice</strong><p>{report.clinical_notice}</p></div></section>

            {isFinal ? <section className="report-signature"><div><span>Reviewed / signed by</span><strong>{finalized.clinician_name}</strong><small>{formatDate(finalized.signed_at)}</small></div><div><span>Report checksum</span><strong className="mono">{finalized.report_checksum_sha256}</strong><small>Finalized report record</small></div>{finalized.clinician_comment ? <div className="report-signature__comment"><span>Reviewer comment</span><p>{finalized.clinician_comment}</p></div> : null}</section> : null}

            <footer className="report-print-footer"><span>NeuroGlioma AI</span><span>{REPORT_UI_VERSION}</span></footer>
          </article>

          <aside className="report-side no-print">
            <section className="report-side-card glass-panel"><span className="eyebrow">REPORT STATUS</span><h3>{isFinal ? 'Finalized report' : readyToFinalize ? 'Ready for clinician sign-off' : 'Preview requires action'}</h3><p>{isFinal ? 'This report has been finalized.' : readyToFinalize ? 'Review the report before sign-off.' : 'Complete the required items before finalization.'}</p>{blockers.length ? <div className="report-blockers">{blockers.map((item)=><span key={item}><Icon name="alert" size={14}/>{item.replaceAll('_',' ')}</span>)}</div> : <div className="report-ready-line"><Icon name="check" size={15}/>No report blockers</div>}</section>

            {!isFinal ? <form className="report-side-card report-signoff-card glass-panel" onSubmit={handleFinalize}><span className="eyebrow">CLINICIAN SIGN-OFF</span><h3>Finalize assessment</h3><label><span>Clinician name</span><input value={clinicianName} onChange={(e)=>setClinicianName(e.target.value)} placeholder="Reviewer name" required minLength={2}/></label><label><span>Review comment <small>optional</small></span><textarea value={clinicianComment} onChange={(e)=>setClinicianComment(e.target.value)} placeholder="Add a concise review note" rows={5}/></label><div className="report-signoff-note"><Icon name="lock" size={15}/><p>Finalization records reviewer attribution and the report timestamp.</p></div><button className="report-primary-button report-primary-button--full" disabled={!readyToFinalize || signing || clinicianName.trim().length < 2}>{signing ? 'Finalizing…' : 'Finalize report'}<Icon name="arrow" size={15}/></button></form> : <section className="report-side-card glass-panel"><span className="eyebrow">SIGN-OFF</span><div className="report-final-seal"><Icon name="check" size={22}/></div><h3>{finalized.clinician_name}</h3><p>{formatDate(finalized.signed_at)}</p><small className="report-hash mono">{finalized.report_checksum_sha256}</small></section>}

            <section className="report-side-card glass-panel"><span className="eyebrow">EXPORT</span><h3>Print or save</h3><button className="report-secondary-button report-secondary-button--full" onClick={() => window.print()}><Icon name="print" size={15}/>Print / Save PDF</button></section>
          </aside>
        </motion.div>

        <AnimatePresence>{notice ? <motion.div className="report-toast no-print" initial={{opacity:0,y:16}} animate={{opacity:1,y:0}} exit={{opacity:0,y:12}}><Icon name="check" size={16}/>{notice}</motion.div> : null}{error ? <motion.div className="report-toast report-toast--error no-print" initial={{opacity:0,y:16}} animate={{opacity:1,y:0}}><Icon name="alert" size={16}/>{error}</motion.div> : null}</AnimatePresence>
      </main>
    </MotionConfig>
  );
}
