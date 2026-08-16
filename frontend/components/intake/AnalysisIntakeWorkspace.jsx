'use client';

import Link from 'next/link';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { AnimatePresence, motion, MotionConfig, useReducedMotion } from 'motion/react';

import {
  confirmBrainScope,
  confirmDicomSeriesSequence,
  confirmNiftiSequenceMapping,
  createUnifiedIntake,
  enqueueSegmentationJob,
  fetchClassifierRuntimeStatus,
  fetchCurrentCapabilities,
  fetchCurrentStudyQc,
  fetchCurrentClassifierResult,
  fetchSegmentationJob,
  fetchStudySeries,
  prepareSegmentationGeometry,
  prepareSegmentationModelInput,
  prepareSegmentationVolumes,
  runStudyClassifier,
  routeStudyCapabilities,
  runAnatomicalLocalization,
  runSegmentationPreflight,
  runStudyQc,
  runTumorQuantification,
  uploadStudySource,
  updateStudyIntakeContext,
} from '@/lib/api';
import { getActiveCase, patchActiveCase, setActiveCase } from '@/lib/activeCase';

const SYMPTOMS = [
  ['headache', 'Headache'], ['seizure', 'Seizure'], ['weakness', 'Weakness'],
  ['vision_change', 'Vision change'], ['speech_change', 'Speech change'],
  ['cognitive_change', 'Cognitive change'], ['other', 'Other'],
];
const SEQUENCES = ['T1C', 'T1', 'T2', 'FLAIR'];
const DICOM_SEQUENCE_OPTIONS = ['T1', 'T1C', 'T2', 'FLAIR', 'OTHER', 'NOT_USABLE'];

const CAPABILITY_LABELS = {
  two_d_classification: '2D classification',
  gradcam_2d: '2D explainability',
  three_d_segmentation: '3D segmentation',
  physical_volume: 'Tumor measurements',
  anatomical_localization: 'Anatomical localization',
};

const CAPABILITY_MESSAGES = {
  two_d_classification: 'Available for supported standalone 2D MRI images.',
  gradcam_2d: 'Available when the 2D classifier is supported for the current input.',
  three_d_segmentation: 'The study can proceed to 3D segmentation.',
  physical_volume: 'Available after a valid segmentation is generated.',
  anatomical_localization: 'Available after segmentation and spatial registration complete.',
};

const QC_MESSAGE_LABELS = {
  DICOM_BRAIN_SCOPE_UNVERIFIED: 'Confirm brain MRI scope',
  DICOM_PIXEL_PRIVACY_NOT_FORMALLY_VALIDATED: 'Review image privacy',
  SEQUENCE_DETECTION_HEURISTIC_NOT_CLINICALLY_VALIDATED: 'Confirm detected sequences',
  NIFTI_BRAIN_SCOPE_UNVERIFIED: 'Confirm brain MRI scope',
  NIFTI_SEQUENCE_MAPPING_REQUIRES_CONFIRMATION: 'Confirm MRI sequence mapping',
};

function friendlyQcMessage(value) {
  return QC_MESSAGE_LABELS[value] || String(value || '').replaceAll('_', ' ').toLowerCase().replace(/^./, (c) => c.toUpperCase());
}

const CAPABILITY_STATE_MESSAGES = {
  two_d_classification: {
    eligible: 'Ready for standalone 2D GBM classification.',
    review_required: 'Resolve the highlighted clinical or image review item before running the classifier.',
    blocked: 'The current input is outside the supported standalone 2D classifier path.',
  },
  gradcam_2d: {
    eligible: '2D explainability is available for this classifier result.',
    deferred: 'Available after a successful 2D classification run.',
    review_required: 'Available after the 2D input review is resolved.',
    blocked: '2D explainability is not available for the current input.',
  },
  three_d_segmentation: {
    eligible: 'Ready for multimodal 3D segmentation.',
    deferred: '3D segmentation is waiting for required preparation.',
    review_required: 'Review the required 3D MRI sequences before segmentation.',
    blocked: 'Requires aligned T1C, T1, T2 and FLAIR volumetric MRI; standalone images remain 2D.',
  },
  physical_volume: {
    eligible: 'Physical tumor measurements are available.',
    deferred: 'Available after a valid 3D segmentation is generated.',
    blocked: 'Requires valid volumetric spatial metadata and a 3D segmentation.',
  },
  anatomical_localization: {
    eligible: 'Atlas-derived anatomical localization is available.',
    deferred: 'Available after segmentation and validated standard-space registration.',
    blocked: 'Requires volumetric MRI, segmentation and validated spatial registration.',
  },
};

const REVIEW_REASON_LABELS = {
  AGE_SCOPE_UNVERIFIED: 'Patient age is required to confirm the adult V1 workflow.',
  BRAIN_SCOPE_CONFIRMATION_REQUIRED: 'Confirm that the upload is a brain MRI study.',
  BRAIN_SCOPE_UNVERIFIED_FOR_RASTER: 'Confirm that the standalone image is a brain MRI.',
  RASTER_LOW_RESOLUTION: 'Image resolution requires manual review before inference.',
  RASTER_LOW_CONTRAST: 'Image contrast requires manual review before inference.',
  RASTER_EXTREME_ASPECT_RATIO: 'Image geometry requires manual review before inference.',
};

function capabilityMessage(key, value) {
  const state = String(value?.state || '');
  if (value?.user_message && state !== 'blocked') return value.user_message;
  return CAPABILITY_STATE_MESSAGES[key]?.[state] || CAPABILITY_MESSAGES[key] || 'Review the current analysis availability.';
}

function reviewReasonLabel(value) {
  return REVIEW_REASON_LABELS[value] || friendlyQcMessage(value);
}

function decisionCopy(result) {
  const state = String(result?.decision_state || 'indeterminate');
  if (state === 'gbm_suspected') {
    return {
      label: 'GBM suspected',
      tone: 'warn',
      detail: 'This MRI shows features that are concerning for glioblastoma (GBM) on the current model. Urgent neuroradiology / neuro-oncology review is advised.',
      meaning: 'The AI pattern is suspicious for a high-grade glioma. This is a triage-style decision support result, not a confirmed diagnosis.',
      action: 'Escalate for specialist review, correlate with the full MRI study, and proceed with the appropriate diagnostic workup.',
      severity: 'High-priority review',
    };
  }
  if (state === 'gbm_not_suspected') {
    return {
      label: 'GBM not suspected',
      tone: 'good',
      detail: 'The current model does not detect a strong imaging pattern for GBM in this 2D MRI image.',
      meaning: 'This result lowers suspicion for GBM on this image, but it does not exclude another tumor, stroke, infection, or other intracranial abnormality.',
      action: 'Review the image in clinical context. If symptoms or other findings remain concerning, continue standard neuroradiology follow-up.',
      severity: 'Routine specialist review',
    };
  }
  return {
    label: 'Indeterminate',
    tone: 'warn',
    detail: 'The system cannot provide a reliable GBM decision for this image.',
    meaning: 'Image quality, scope, or model uncertainty prevents a confident automated interpretation.',
    action: 'Use manual radiology review and, if needed, repeat the workflow with a better-quality MRI input.',
    severity: 'Manual review required',
  };
}

function probabilityBand(probability) {
  if (probability == null || Number.isNaN(Number(probability))) return 'Unavailable';
  const value = Number(probability);
  if (value >= 0.75) return 'High';
  if (value >= 0.4) return 'Intermediate';
  return 'Low';
}

function Icon({ name, size = 18 }) {
  const common = { width:size, height:size, viewBox:'0 0 24 24', fill:'none', stroke:'currentColor', strokeWidth:1.8, strokeLinecap:'round', strokeLinejoin:'round', 'aria-hidden':true };
  const paths = {
    arrow:<><path d="M5 12h14"/><path d="m13 6 6 6-6 6"/></>,
    back:<><path d="m15 18-6-6 6-6"/></>,
    upload:<><path d="M12 16V4"/><path d="m7 9 5-5 5 5"/><path d="M5 20h14"/></>,
    scan:<><path d="M7 3H5a2 2 0 0 0-2 2v2M17 3h2a2 2 0 0 1 2 2v2M7 21H5a2 2 0 0 1-2-2v-2M17 21h2a2 2 0 0 0 2-2v-2"/><circle cx="12" cy="12" r="4"/></>,
    check:<path d="m5 12 4 4L19 6"/>,
    shield:<><path d="M12 3 5 6v5c0 4.5 2.9 8.2 7 10 4.1-1.8 7-5.5 7-10V6l-7-3Z"/><path d="m9 12 2 2 4-4"/></>,
    alert:<><path d="M12 3 2.7 20h18.6L12 3Z"/><path d="M12 9v4M12 17h.01"/></>,
    brain:<><path d="M9.5 4A3.5 3.5 0 0 0 6 7.5v.3A3.5 3.5 0 0 0 4.5 14 3.5 3.5 0 0 0 8 17.5V19a2 2 0 0 0 4 0V7.5A3.5 3.5 0 0 0 9.5 4Z"/><path d="M14.5 4A3.5 3.5 0 0 1 18 7.5v.3a3.5 3.5 0 0 1 1.5 6.2 3.5 3.5 0 0 1-3.5 3.5V19a2 2 0 0 1-4 0V7.5A3.5 3.5 0 0 1 14.5 4Z"/></>,
  };
  return <svg {...common}>{paths[name] || paths.scan}</svg>;
}

function StepRail({ stage, twoD = false }) {
  const order = ['details','upload','qc','eligible','processing','complete'];
  const current = Math.max(0, order.indexOf(stage));
  const steps = [['01','Intake'],['02','MRI upload'],['03','QC & mapping'],['04','AI eligibility'],['05','Processing'],['06',twoD?'Result':'Viewer']];
  return <div className="intake-step-rail">{steps.map((item,index)=><div key={item[0]} className={index < current ? 'intake-step intake-step--done' : index === current ? 'intake-step intake-step--active':'intake-step'}><span>{index < current ? <Icon name="check" size={13}/> : item[0]}</span><strong>{item[1]}</strong>{index < steps.length-1 ? <i/>:null}</div>)}</div>;
}

function StatusCard({ title, value, tone='neutral', detail }) {
  return <div className={`intake-status-card intake-status-card--${tone}`}><span>{title}</span><strong>{value}</strong>{detail ? <small>{detail}</small>:null}</div>;
}

function BusyMark() {
  return <span className="button-working-mark" aria-hidden="true"/>;
}

export default function AnalysisIntakeWorkspace() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const reduceMotion = useReducedMotion();
  const pollRef = useRef(null);
  const [stage,setStage] = useState('details');
  const [busy,setBusy] = useState(false);
  const [navigating,setNavigating] = useState(false);
  const [error,setError] = useState('');
  const [notice,setNotice] = useState('');
  const [studyUuid,setStudyUuid] = useState(null);
  const [caseReference,setCaseReference] = useState(null);
  const [upload,setUpload] = useState(null);
  const [sourceFormat,setSourceFormat] = useState(null);
  const [qc,setQc] = useState(null);
  const [capabilities,setCapabilities] = useState(null);
  const [series,setSeries] = useState([]);
  const [brainConfirmed,setBrainConfirmed] = useState(false);
  const [mapping,setMapping] = useState({ t1:'', t1c:'', t2:'', flair:'' });
  const [seriesMapping,setSeriesMapping] = useState({});
  const [job,setJob] = useState(null);
  const [processingLog,setProcessingLog] = useState([]);
  const [classifierRuntime,setClassifierRuntime] = useState(null);
  const [classificationResult,setClassificationResult] = useState(null);
  const [scopeAge,setScopeAge] = useState('');
  const [form,setForm] = useState({
    case_reference:'', patient_name:'', age_years:'', sex:'unknown',
    mri_date:new Date().toISOString().slice(0,10), symptoms:[], symptom_duration:'',
    prior_treatment:false, clinical_notes:'',
  });

  useEffect(()=>{
    if (searchParams.get('resume') !== '1') return;
    const active = getActiveCase();
    if (!active?.studyUuid) return;
    setStudyUuid(active.studyUuid); setCaseReference(active.caseReference); setSourceFormat(active.sourceFormat || null);
    const nextStage = ['viewer_ready','classification_ready'].includes(active.stage) ? 'complete' : (active.stage || 'upload');
    setStage(nextStage);
    setNotice(`Resumed ${active.caseReference}. Restoring the latest workflow state…`);
    (async()=>{
      try {
        if (['qc','eligible','processing','viewer_ready','classification_ready'].includes(active.stage)) {
          const currentQc=await fetchCurrentStudyQc(active.studyUuid); setQc(currentQc);
          setBrainConfirmed(true);
        }
        if (['eligible','processing','viewer_ready','classification_ready'].includes(active.stage)) {
          const currentCapabilities=await fetchCurrentCapabilities(active.studyUuid); setCapabilities(currentCapabilities);
          if (active.sourceFormat === 'image') {
            try { setClassifierRuntime(await fetchClassifierRuntimeStatus()); } catch {}
          }
        }
        if (active.stage === 'classification_ready') {
          setClassificationResult(await fetchCurrentClassifierResult(active.studyUuid));
        }
        setNotice(`Resumed ${active.caseReference}. You can continue from the last saved step.`);
      } catch {
        setNotice(`Resumed ${active.caseReference}. Recheck the current step before continuing.`);
      }
    })();
  },[searchParams]);

  useEffect(()=>()=>{ if (pollRef.current) window.clearTimeout(pollRef.current); },[]);

  const volumes = useMemo(()=>qc?.checks?.volumes || [],[qc]);
  const threeDEligible = capabilities?.capabilities?.three_d_segmentation?.state === 'eligible';
  const twoDState = capabilities?.capabilities?.two_d_classification?.state || null;
  const twoDEligible = twoDState === 'eligible';
  const twoDReviewReasons = capabilities?.capabilities?.two_d_classification?.reasons || [];
  const isTwoD = sourceFormat === 'image';
  const resultSummary = useMemo(() => decisionCopy(classificationResult), [classificationResult]);
  const requiresNiftiMapping = String(qc?.checks?.sequence_mapping_status || '').includes('REQUIRES_CONFIRMATION');
  const niftiMappingValues = [mapping.t1, mapping.t1c, mapping.t2, mapping.flair];
  const niftiMappingComplete = !requiresNiftiMapping || (niftiMappingValues.every(Boolean) && new Set(niftiMappingValues).size === 4);
  const dicomMappingComplete = !series.length || series.every((item) => Boolean(seriesMapping[item.id]));
  const qcReadyForRouting = Boolean(brainConfirmed && niftiMappingComplete && dicomMappingComplete && qc?.qc_status !== 'fail');

  function updateForm(key,value){ setForm((current)=>({...current,[key]:value})); }
  function toggleSymptom(value){ setForm((current)=>({...current,symptoms:current.symptoms.includes(value)?current.symptoms.filter((x)=>x!==value):[...current.symptoms,value]})); }
  function addLog(label,status='done'){ setProcessingLog((current)=>[...current,{label,status,at:Date.now()}]); }

  async function createCase(event){
    event.preventDefault(); setBusy(true); setError('');
    try {
      const payload={
        ...form,
        case_reference:form.case_reference.trim() || null,
        patient_name:form.patient_name.trim() || null,
        age_years:form.age_years ? Number(form.age_years):null,
        symptom_duration:form.symptom_duration.trim() || null,
        clinical_notes:form.clinical_notes.trim() || null,
      };
      const result=await createUnifiedIntake(payload);
      setStudyUuid(result.study_uuid); setCaseReference(result.case_reference); setStage('upload');
      setActiveCase({studyUuid:result.study_uuid,caseReference:result.case_reference,stage:'upload',sourceFormat:null});
      setNotice(result.patient_reused ? `Case ${result.case_reference} resumed with a new MRI assessment.` : `Case ${result.case_reference} is ready for MRI upload.`);
    } catch(reason){ setError(reason?.message || 'Could not create the analysis intake.'); }
    finally{ setBusy(false); }
  }

  async function uploadAndQc(){
    if (!upload || !studyUuid) return;
    setBusy(true); setError(''); setNotice('');
    try {
      const uploaded=await uploadStudySource(studyUuid,upload);
      const result=await runStudyQc(studyUuid);
      setQc(result); setStage('qc'); setSourceFormat(uploaded.source_format);
      patchActiveCase({stage:'qc',sourceFormat:uploaded.source_format});
      if (uploaded.source_format === 'dicom') {
        const list=await fetchStudySeries(studyUuid);
        setSeries(list.series || []);
        const initial={};
        for (const item of list.series || []) initial[item.id]=item.confirmed_sequence || item.detected_sequence || 'OTHER';
        setSeriesMapping(initial);
      }
      setNotice(`MRI received as ${String(uploaded.source_format).toUpperCase()}. Quality and spatial checks completed before analysis.`);
    } catch(reason){ setError(reason?.message || 'MRI upload or QC failed.'); }
    finally{ setBusy(false); }
  }

  async function confirmAndRoute(){
    if (!studyUuid || !qc) return;
    setBusy(true); setError('');
    try {
      if (!brainConfirmed) throw new Error('Confirm that the uploaded study is a brain MRI before capability routing.');
      await confirmBrainScope(studyUuid,true);
      if (String(qc?.checks?.sequence_mapping_status || '').includes('REQUIRES_CONFIRMATION')) {
        const values=[mapping.t1,mapping.t1c,mapping.t2,mapping.flair];
        if (values.some((value)=>value==='')) throw new Error('Map each T1, T1C, T2 and FLAIR channel to a distinct NIfTI volume.');
        if (new Set(values).size !== 4) throw new Error('Each MRI sequence must use a different NIfTI volume.');
        await confirmNiftiSequenceMapping(studyUuid,{t1:Number(mapping.t1),t1c:Number(mapping.t1c),t2:Number(mapping.t2),flair:Number(mapping.flair)});
      }
      if (series.length) {
        for (const item of series) {
          const selected=seriesMapping[item.id];
          if (selected && selected !== item.confirmed_sequence) await confirmDicomSeriesSequence(item.id,selected);
        }
        const refreshed=await runStudyQc(studyUuid); setQc(refreshed);
      }
      const routed=await routeStudyCapabilities(studyUuid);
      setCapabilities(routed); setStage('eligible'); patchActiveCase({stage:'eligible'});
      if (sourceFormat === 'image') {
        try { setClassifierRuntime(await fetchClassifierRuntimeStatus()); } catch { setClassifierRuntime(null); }
        const state = routed.capabilities?.two_d_classification?.state;
        if (state === 'eligible') setNotice('The standalone image passed the supported 2D input checks and is ready for classification.');
        else setNotice('One review item must be resolved before 2D classification can run.');
      } else if (routed.capabilities?.three_d_segmentation?.state !== 'eligible') {
        setNotice('The study was routed safely. Review the items below before 3D analysis.');
      } else setNotice('Required 3D MRI inputs passed capability routing and are ready for 3D analysis.');
    } catch(reason){ setError(reason?.message || 'Sequence/scope confirmation or capability routing failed.'); }
    finally{ setBusy(false); }
  }

  async function resolveAdultScope(){
    const age=Number(scopeAge);
    if (!studyUuid || !Number.isFinite(age) || age < 18 || age > 100) { setError('Enter a valid adult age from 18 to 100.'); return; }
    setBusy(true); setError('');
    try {
      await updateStudyIntakeContext(studyUuid,{age_years:age});
      const routed=await routeStudyCapabilities(studyUuid);
      setCapabilities(routed);
      try { setClassifierRuntime(await fetchClassifierRuntimeStatus()); } catch { setClassifierRuntime(null); }
      setNotice('Adult workflow scope confirmed. Classification eligibility has been refreshed.');
    } catch(reason){ setError(reason?.message || 'Could not update the clinical context.'); }
    finally{ setBusy(false); }
  }

  async function run2DClassification(){
    if (!studyUuid || !twoDEligible || busy) return;
    setBusy(true); setError(''); setNotice(''); setStage('processing'); setProcessingLog([]); patchActiveCase({stage:'processing'});
    addLog('Standalone 2D MRI input confirmed','done');
    addLog('EfficientNetV2-S ensemble inference','active');
    try {
      const result=await runStudyClassifier(studyUuid);
      setClassificationResult(result);
      setProcessingLog((items)=>items.map((item)=>item.status==='active'?{...item,status:'done'}:item));
      addLog('Calibrated GBM probability and safety state generated');
      setStage('complete'); patchActiveCase({stage:'classification_ready'});
    } catch(reason){
      setError(reason?.message || '2D classification could not be completed.');
      setStage('eligible'); patchActiveCase({stage:'eligible'});
    } finally { setBusy(false); }
  }

  async function startProcessing(){
    if (!studyUuid || !threeDEligible) return;
    setBusy(true); setError(''); setStage('processing'); setProcessingLog([]); patchActiveCase({stage:'processing'});
    try {
      await runSegmentationPreflight(studyUuid); addLog('3D segmentation preflight');
      await prepareSegmentationVolumes(studyUuid); addLog('Volumes loaded + orientation/alignment validation');
      await prepareSegmentationGeometry(studyUuid); addLog('Registration + 1 mm model geometry');
      await prepareSegmentationModelInput(studyUuid); addLog('MONAI model input normalization');
      const queued=await enqueueSegmentationJob(studyUuid); setJob(queued); addLog('SegResNet background job queued','active');
      setBusy(false); pollJob(queued.job_uuid);
    } catch(reason){ setBusy(false); setError(reason?.message || 'AI preparation failed before inference.'); setStage('eligible'); patchActiveCase({stage:'eligible'}); }
  }

  async function pollJob(jobUuid){
    try {
      const current=await fetchSegmentationJob(jobUuid); setJob(current);
      if (current.status === 'complete') {
        setProcessingLog((items)=>items.map((item)=>item.status==='active'?{...item,status:'done'}:item));
        addLog('WT / TC / ET segmentation generated');
        try { await runTumorQuantification(studyUuid); addLog('Physical tumor quantification'); } catch(reason){ addLog(`Quantification unavailable: ${reason?.message || 'review required'}`,'warning'); }
        try { await runAnatomicalLocalization(studyUuid); addLog('Atlas-based anatomical localization'); } catch(reason){ addLog(`Localization unavailable: ${reason?.message || 'review required'}`,'warning'); }
        setStage('complete'); patchActiveCase({stage:'viewer_ready'}); return;
      }
      if (current.status === 'failed') { setError(current.last_error_code ? `Segmentation job failed: ${current.last_error_code}` : 'Segmentation background job failed.'); return; }
      pollRef.current=window.setTimeout(()=>pollJob(jobUuid),4000);
    } catch(reason){ setError(reason?.message || 'Could not read the background segmentation job.'); }
  }

  async function openViewer(){
    if (navigating) return;
    setNavigating(true); setError('');
    // These calls are idempotent. They also repair legacy completed studies
    // whose segmentation reference was written before the ORM UUID existed.
    try { await runTumorQuantification(studyUuid); } catch {}
    try { await runAnatomicalLocalization(studyUuid); } catch {}
    patchActiveCase({stage:'viewer_ready'});
    router.push('/viewer/current');
  }

  return (
    <MotionConfig reducedMotion="user">
      <main className="intake-shell">
        <div className="ambient ambient--one"/><div className="ambient ambient--two"/>
        {(busy || navigating) ? <div className="operation-progress" role="status" aria-live="polite"><span/><small>{navigating ? 'Opening viewer…' : (isTwoD && stage==='processing' ? 'Running 2D analysis…' : 'Processing…')}</small></div> : null}
        <header className="intake-topbar"><Link href="/" className="intake-back"><Icon name="back"/>Home</Link><div className="intake-brand"><span className="intake-brand-mark"><Icon name="brain"/></span><div><span className="eyebrow">NEUROGLIOMA AI</span><strong>New MRI Analysis</strong></div></div><div className="intake-security"><Icon name="shield"/><span>MRI analysis workflow</span></div></header>
        <StepRail stage={stage} twoD={isTwoD}/>
        <section className="intake-stage-wrap">
          <AnimatePresence mode="wait">
            {stage==='details' ? <motion.div key="details" className="intake-stage-card glass-panel" initial={{opacity:0,y:10}} animate={{opacity:1,y:0}} exit={{opacity:0,y:-8}}>
              <div className="intake-stage-heading"><div><span className="eyebrow">STEP 01 · CLINICAL CONTEXT</span><h1>Start a new MRI analysis</h1><p>Enter the clinical context for this MRI assessment.</p></div><span className="intake-stage-icon"><Icon name="brain" size={26}/></span></div>
              <form onSubmit={createCase} className="intake-form">
                <div className="field-grid field-grid--3"><label><span>Case reference <small>optional</small></span><input value={form.case_reference} onChange={(e)=>updateForm('case_reference',e.target.value)} placeholder="Auto-generate if blank"/></label><label><span>Patient name <small>optional</small></span><input value={form.patient_name} onChange={(e)=>updateForm('patient_name',e.target.value)} placeholder="De-identified / local use"/></label><label><span>MRI date</span><input type="date" required value={form.mri_date} onChange={(e)=>updateForm('mri_date',e.target.value)}/></label></div>
                <div className="field-grid field-grid--3"><label><span>Age <small>required for adult scope</small></span><input type="number" min="18" max="100" required value={form.age_years} onChange={(e)=>updateForm('age_years',e.target.value)} placeholder="18–100"/></label><label><span>Sex</span><select value={form.sex} onChange={(e)=>updateForm('sex',e.target.value)}><option value="unknown">Unknown</option><option value="female">Female</option><option value="male">Male</option><option value="other">Other</option></select></label><label className="toggle-field"><span>Prior brain tumor treatment?</span><button type="button" className={form.prior_treatment?'yes':'no'} onClick={()=>updateForm('prior_treatment',!form.prior_treatment)}><i/>{form.prior_treatment?'Yes — outside V1 scope':'No'}</button></label></div>
                <div className="field-block"><span>Presenting symptoms</span><div className="symptom-chips">{SYMPTOMS.map(([value,label])=><button type="button" key={value} className={form.symptoms.includes(value)?'active':''} onClick={()=>toggleSymptom(value)}>{label}</button>)}</div></div>
                <div className="field-grid field-grid--2"><label><span>Symptom duration</span><input value={form.symptom_duration} onChange={(e)=>updateForm('symptom_duration',e.target.value)} placeholder="e.g. 3 weeks"/></label><label><span>Clinical note <small>optional</small></span><input value={form.clinical_notes} onChange={(e)=>updateForm('clinical_notes',e.target.value)} placeholder="Brief context only"/></label></div>
                <div className="intake-form-foot"><div/><motion.button className={busy ? "intake-primary is-working" : "intake-primary"} type="submit" disabled={busy} aria-busy={busy} whileTap={reduceMotion?undefined:{scale:.98}}>{busy ? <><BusyMark/>Creating analysis…</> : <>Create analysis<Icon name="arrow"/></>}</motion.button></div>
              </form>
            </motion.div> : null}

            {stage==='upload' ? <motion.div key="upload" className="intake-stage-card glass-panel" initial={{opacity:0,y:10}} animate={{opacity:1,y:0}}>
              <div className="intake-stage-heading"><div><span className="eyebrow">STEP 02 · MRI INTAKE</span><h1>Upload brain MRI</h1><p>Case <strong>{caseReference}</strong> is ready for MRI upload.</p></div><span className="intake-stage-icon"><Icon name="upload" size={26}/></span></div>
              <label className={upload?'upload-dropzone upload-dropzone--ready':'upload-dropzone'}><input type="file" accept=".zip,.nii,.gz,.dcm,.dicom,.ima,.jpg,.jpeg,.png" onChange={(e)=>setUpload(e.target.files?.[0] || null)}/><span className="upload-orbit"><Icon name="upload" size={27}/></span><strong>{upload?upload.name:'Drop or choose one MRI package'}</strong><p>{upload?`${(upload.size/1024/1024).toFixed(1)} MB selected`:'DICOM ZIP, multi-volume NIfTI ZIP, or supported standalone MRI image'}</p><small>Format detection is automatic. Do not mix DICOM and NIfTI in one archive.</small></label>
              <div className="intake-action-row"><Link href="/" className="intake-secondary">Cancel</Link><button className={busy ? "intake-primary is-working" : "intake-primary"} disabled={!upload || busy} aria-busy={busy} onClick={uploadAndQc}>{busy ? <><BusyMark/>Uploading & validating…</> : <>Upload & run MRI QC<Icon name="arrow"/></>}</button></div>
            </motion.div> : null}

            {stage==='qc' ? <motion.div key="qc" className="intake-stage-card intake-stage-card--wide glass-panel" initial={{opacity:0,y:10}} animate={{opacity:1,y:0}}>
              <div className="intake-stage-heading"><div><span className="eyebrow">STEP 03 · QUALITY & SEQUENCE REVIEW</span><h1>Confirm imaging context</h1><p>AI processing remains blocked until brain scope and required sequence mapping are explicit.</p></div><span className="intake-stage-icon"><Icon name="scan" size={26}/></span></div>
              <div className="intake-status-grid"><StatusCard title="QC status" value={String(qc?.qc_status || 'pending').toUpperCase()} tone={qc?.qc_status==='fail'?'bad':qc?.qc_status==='pass'?'good':'warn'}/><StatusCard title="Volumes" value={String(qc?.checks?.volume_count ?? series.length ?? 0)} detail={series.length?'DICOM series':'NIfTI volumes'}/><StatusCard title="Manual review" value={qc?.manual_review_required?'REQUIRED':'NOT REQUIRED'} tone={qc?.manual_review_required?'warn':'good'}/></div>
              {(qc?.fail_reasons?.length || qc?.warnings?.length || qc?.partial_reasons?.length) ? <div className="qc-messages">{qc?.fail_reasons?.map((x)=><span className="bad" key={x}><Icon name="alert" size={13}/>{friendlyQcMessage(x)}</span>)}{qc?.partial_reasons?.map((x)=><span className="warn" key={x}><Icon name="alert" size={13}/>{friendlyQcMessage(x)}</span>)}{qc?.warnings?.map((x)=><span key={x}>{friendlyQcMessage(x)}</span>)}</div>:null}
              <label className={brainConfirmed?'brain-confirm brain-confirm--active':'brain-confirm brain-confirm--attention'}>
                <input type="checkbox" checked={brainConfirmed} onChange={(e)=>setBrainConfirmed(e.target.checked)}/>
                <span className="brain-confirm__icon"><Icon name="brain"/></span>
                <div className="brain-confirm__copy">
                  <em className="brain-confirm__eyebrow">{brainConfirmed ? 'Confirmation recorded' : 'Required action'}</em>
                  <strong>I confirm this upload is a brain MRI study</strong>
                  <small>{brainConfirmed ? 'Brain imaging scope is confirmed. You can now continue to AI eligibility.' : 'Click anywhere on this bar to confirm the case is a brain MRI study and unlock the next step.'}</small>
                </div>
                <i>{brainConfirmed ? <Icon name="check" size={14}/> : <span className="brain-confirm__pulse-dot" aria-hidden="true"/>}</i>
              </label>
              {volumes.length ? <div className="mapping-panel"><div className="mapping-heading"><div><span className="eyebrow">NIFTI CHANNEL MAPPING</span><h2>Map the four MRI sequences</h2></div><p>Confirm the sequence mapping before analysis.</p></div><div className="mapping-grid">{[['t1c','T1C'],['t1','T1'],['t2','T2'],['flair','FLAIR']].map(([key,label])=><label key={key}><span>{label}</span><select value={mapping[key]} onChange={(e)=>setMapping((m)=>({...m,[key]:e.target.value}))}><option value="">Choose volume</option>{volumes.map((volume)=><option value={volume.volume_index} key={volume.volume_index}>Volume {Number(volume.volume_index)+1} · {(volume.shape||[]).join('×')} · {(volume.zooms||[]).slice(0,3).join('/')} mm</option>)}</select></label>)}</div></div>:null}
              {series.length ? <div className="mapping-panel"><div className="mapping-heading"><div><span className="eyebrow">DICOM SERIES REVIEW</span><h2>Confirm detected sequences</h2></div><p>Confirm the detected MRI sequence labels.</p></div><div className="dicom-series-list">{series.map((item)=><div key={item.id}><div><strong>Series {item.series_number ?? '—'}</strong><span>{item.slice_count} slices</span><small>{item.sequence_metadata?.series_description || item.sequence_metadata?.protocol_name || 'Sequence description unavailable'}</small></div><select value={seriesMapping[item.id] || 'OTHER'} onChange={(e)=>setSeriesMapping((m)=>({...m,[item.id]:e.target.value}))}>{DICOM_SEQUENCE_OPTIONS.map((label)=><option key={label} value={label}>{label}</option>)}</select></div>)}</div></div>:null}
              <div className={`qc-next-step ${qcReadyForRouting ? 'qc-next-step--ready' : 'qc-next-step--attention'}`} role="status" aria-live="polite">
                <span>{qcReadyForRouting ? <Icon name="check" size={15}/> : <Icon name="alert" size={15}/>}</span>
                <div>
                  <strong>{qcReadyForRouting ? 'Ready for AI eligibility' : !brainConfirmed ? '1 required confirmation remaining' : !niftiMappingComplete ? 'Complete the MRI sequence mapping' : !dicomMappingComplete ? 'Review the detected DICOM sequences' : 'Review the highlighted QC item'}</strong>
                  <small>{qcReadyForRouting ? 'All required review items are complete. Continue when ready.' : !brainConfirmed ? 'Select the highlighted brain MRI confirmation bar above to unlock the next step.' : 'Complete the remaining review item before continuing.'}</small>
                </div>
              </div>
              <div className="intake-action-row"><button className="intake-secondary" onClick={()=>setStage('upload')}>Back</button><button className={busy ? "intake-primary is-working" : "intake-primary"} disabled={busy || !qcReadyForRouting} aria-busy={busy} onClick={confirmAndRoute}>{busy ? <><BusyMark/>Checking eligibility…</> : !brainConfirmed ? <>Confirm brain MRI above<Icon name="arrow"/></> : !niftiMappingComplete || !dicomMappingComplete ? <>Complete review to continue<Icon name="arrow"/></> : <>Confirm & check AI eligibility<Icon name="arrow"/></>}</button></div>
            </motion.div> : null}

            {stage==='eligible' ? <motion.div key="eligible" className="intake-stage-card glass-panel" initial={{opacity:0,y:10}} animate={{opacity:1,y:0}}>
              <div className="intake-stage-heading"><div><span className="eyebrow">STEP 04 · AI ELIGIBILITY</span><h1>{isTwoD ? (twoDEligible ? '2D classification is ready' : 'Review required before classification') : (threeDEligible ? '3D analysis is eligible' : 'Review required before 3D analysis')}</h1><p>{isTwoD ? 'The system has selected the standalone 2D MRI pathway for this upload.' : 'Available analysis options are based on the validated volumetric MRI input.'}</p></div><span className={`intake-stage-icon ${(isTwoD?twoDEligible:threeDEligible)?'good':'warn'}`}><Icon name={(isTwoD?twoDEligible:threeDEligible)?'check':'alert'} size={26}/></span></div>
              <div className="capability-list">{Object.entries(capabilities?.capabilities || {}).map(([key,value])=><div key={key}><span className={`capability-state capability-state--${value.state}`}>{String(value.state).replaceAll('_',' ')}</span><div><strong>{CAPABILITY_LABELS[key] || key}</strong><p>{capabilityMessage(key,value)}</p></div></div>)}</div>
              {isTwoD && twoDReviewReasons.length ? <div className="eligibility-resolution"><div><span className="eyebrow">ACTION NEEDED</span><h3>Complete the remaining review</h3>{twoDReviewReasons.map((reason)=><p key={reason}><Icon name="alert" size={14}/>{reviewReasonLabel(reason)}</p>)}</div>{twoDReviewReasons.includes('AGE_SCOPE_UNVERIFIED') ? <div className="eligibility-age"><label><span>Patient age</span><input type="number" min="18" max="100" value={scopeAge} onChange={(e)=>setScopeAge(e.target.value)} placeholder="18–100"/></label><button type="button" className={busy?'intake-primary is-working':'intake-primary'} disabled={busy || !scopeAge} onClick={resolveAdultScope}>{busy?<><BusyMark/>Rechecking…</>:<>Save age & recheck<Icon name="arrow"/></>}</button></div> : <button type="button" className="intake-secondary" onClick={()=>setStage('qc')}>Review image quality</button>}</div> : null}
              {isTwoD && twoDEligible ? <div className={`classifier-readiness ${classifierRuntime?.ready?'classifier-readiness--ready':'classifier-readiness--warn'}`}><div><span className="eyebrow">2D CLASSIFIER</span><strong>{classifierRuntime?.ready === true ? 'Model runtime ready' : classifierRuntime?.ready === false ? 'Model files need attention' : 'Checking model runtime'}</strong><p>{classifierRuntime?.ready === true ? 'The frozen five-fold EfficientNetV2-S ensemble is available on this workstation.' : classifierRuntime?.ready === false ? `The image is eligible, but only ${classifierRuntime.checkpoint_count_available}/${classifierRuntime.checkpoint_count_expected} frozen checkpoints are available.` : 'Verifying the local classifier assets before inference.'}</p></div>{classifierRuntime?.ready === true ? <span className="classifier-ready-badge"><Icon name="check" size={14}/>Ready</span> : <span className="classifier-ready-badge classifier-ready-badge--warn"><Icon name="alert" size={14}/>{classifierRuntime?.ready === false?'Not ready':'Checking'}</span>}</div> : null}
              <div className="intake-action-row"><button className="intake-secondary" onClick={()=>setStage('qc')}>{isTwoD?'Back to image review':'Review mapping'}</button>{isTwoD ? <button className={busy?'intake-primary is-working':'intake-primary'} disabled={!twoDEligible || busy || classifierRuntime?.ready!==true} aria-busy={busy} onClick={run2DClassification}>{busy?<><BusyMark/>Running 2D analysis…</>:<>Run 2D GBM analysis<Icon name="arrow"/></>}</button> : <button className={busy ? "intake-primary is-working" : "intake-primary"} disabled={!threeDEligible || busy} aria-busy={busy} onClick={startProcessing}>{busy ? <><BusyMark/>Preparing analysis…</> : <>Prepare & queue 3D analysis<Icon name="arrow"/></>}</button>}</div>
            </motion.div> : null}

            {stage==='processing' ? <motion.div key="processing" className="intake-stage-card glass-panel" initial={{opacity:0,y:10}} animate={{opacity:1,y:0}}>
              <div className="intake-stage-heading"><div><span className="eyebrow">STEP 05 · AI PROCESSING</span><h1>{isTwoD ? 'Running 2D GBM analysis' : 'Analyzing multimodal MRI'}</h1><p>{isTwoD ? 'The standalone MRI image is being processed by the frozen classifier and safety logic.' : 'The MRI study is being prepared and analyzed.'}</p></div><span className="intake-stage-icon processing"><Icon name="scan" size={26}/></span></div>
              <div className="processing-confirm-strip" role="status" aria-live="polite">
                <div className="processing-confirm-chip processing-confirm-chip--done"><Icon name="check" size={14}/><span>{isTwoD ? 'Brain MRI confirmed' : 'Study confirmed'}</span></div>
                <div className="processing-confirm-chip processing-confirm-chip--done"><Icon name="check" size={14}/><span>{isTwoD ? '2D pathway selected' : 'Eligibility verified'}</span></div>
                <div className="processing-confirm-chip processing-confirm-chip--active"><BusyMark/><span>{isTwoD ? 'Classifier running' : 'AI processing active'}</span></div>
              </div>
              <div className="processing-visual"><div className="processing-radar"><span/><i/><b/></div><div><strong>{isTwoD ? '2D classifier is running' : (job?.status==='running'?'MRI analysis is running':job?.status==='queued'?'Analysis queued':'Preparing MRI data')}</strong><small>{isTwoD ? 'Calculating calibrated probability and safety state' : (job ? `Processing attempt ${job.attempts}/${job.max_attempts}` : 'Preparing analysis')}</small></div></div>
              <div className="processing-log">{processingLog.map((item,index)=><div key={`${item.at}-${index}`} className={`processing-log-row processing-log-row--${item.status}`}><span>{item.status==='done'?<Icon name="check" size={13}/>:item.status==='warning'?<Icon name="alert" size={13}/>:<i/>}</span><strong>{item.label}</strong></div>)}</div>
            </motion.div> : null}

            {stage==='complete' ? (isTwoD ? <motion.div key="complete-2d" className="intake-stage-card classification-result glass-panel" initial={{opacity:0,scale:.985}} animate={{opacity:1,scale:1}}>
              <div className={`classification-result__status classification-result__status--${resultSummary.tone}`}><span className="eyebrow">2D GBM ASSESSMENT</span><h1>{resultSummary.label}</h1><p>{resultSummary.detail}</p></div>
              <div className="classification-result__grid"><div><span>Estimated GBM likelihood</span><strong>{classificationResult?.calibrated_probability_gbm == null ? '—' : `${(Number(classificationResult.calibrated_probability_gbm)*100).toFixed(1)}%`}</strong><small>Model-estimated likelihood for the GBM class on this image</small></div><div><span>Clinical priority</span><strong>{resultSummary.severity}</strong><small>Simple triage meaning for the current result</small></div><div><span>Technical quality</span><strong>{String(classificationResult?.qc_state || '—').replaceAll('_',' ')}</strong><small>Safety / quality state used before giving this output</small></div><div><span>Likelihood band</span><strong>{probabilityBand(classificationResult?.calibrated_probability_gbm)}</strong><small>Low, intermediate, or high model suspicion</small></div><div><span>Input type</span><strong>Standalone 2D MRI</strong><small>Only a single MRI image was analyzed, not a full 3D study</small></div><div><span>Next clinical step</span><strong>{resultSummary.tone === 'good' ? 'Review & document' : resultSummary.tone === 'warn' ? 'Specialist review' : 'Manual review'}</strong><small>Suggested action for the clinician workflow</small></div></div>
              <div className="classification-summary-panels"><div className="classification-summary-panel"><span className="eyebrow">Clinical interpretation</span><h3>What this means</h3><p>{resultSummary.meaning}</p></div><div className="classification-summary-panel"><span className="eyebrow">Recommended action</span><h3>What to do next</h3><p>{resultSummary.action}</p></div></div>
              {classificationResult?.safety_reason_codes?.length ? <div className="classification-safety"><Icon name="shield" size={17}/><div><strong>Safety review</strong><p>{classificationResult.safety_reason_codes.map(reviewReasonLabel).join(' · ')}</p></div></div> : null}
              <div className="classification-result__notice"><Icon name="shield" size={17}/><p>AI-assisted imaging assessment only. Use with radiology review and clinical correlation. This result does not confirm GBM and does not rule out other intracranial disease.</p></div>
              <div className="intake-action-row"><Link href="/analysis/new" className="intake-secondary">New analysis</Link><Link href="/report/current" className="intake-primary">Open report <Icon name="arrow"/></Link></div>
            </motion.div> : <motion.div key="complete" className="intake-stage-card intake-complete glass-panel" initial={{opacity:0,scale:.985}} animate={{opacity:1,scale:1}}>
              <div className="intake-success-orbit"><span><Icon name="check" size={30}/></span></div><span className="eyebrow">ANALYSIS READY</span><h1>{caseReference || getActiveCase()?.caseReference}</h1><p>The MRI analysis is ready for review.</p><div className="complete-badges"><span><Icon name="scan" size={15}/>3-plane MRI</span><span>WT / TC / ET</span><span><Icon name="shield" size={15}/>Review ready</span></div><button className={navigating ? "intake-primary intake-primary--large is-working" : "intake-primary intake-primary--large"} disabled={navigating} aria-busy={navigating} onClick={openViewer}>{navigating ? <><BusyMark/>Opening viewer…</> : <>Open clinical viewer <Icon name="arrow"/></>}</button>
            </motion.div>) : null}
          </AnimatePresence>
          {notice ? <div className="intake-banner intake-banner--info" role="status" aria-live="polite">{notice}</div>:null}
          {error ? <div className="intake-banner intake-banner--error" role="alert"><Icon name="alert" size={15}/><span>{error}</span></div>:null}
        </section>
        <footer className="intake-footer"><span>NeuroGlioma AI</span></footer>
      </main>
    </MotionConfig>
  );
}
