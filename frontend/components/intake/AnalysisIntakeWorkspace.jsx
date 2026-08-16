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
  fetchSegmentationJob,
  fetchStudySeries,
  prepareSegmentationGeometry,
  prepareSegmentationModelInput,
  prepareSegmentationVolumes,
  routeStudyCapabilities,
  runAnatomicalLocalization,
  runSegmentationPreflight,
  runStudyQc,
  runTumorQuantification,
  uploadStudySource,
} from '@/lib/api';
import { getActiveCase, patchActiveCase, setActiveCase } from '@/lib/activeCase';

const SYMPTOMS = [
  ['headache', 'Headache'], ['seizure', 'Seizure'], ['weakness', 'Weakness'],
  ['vision_change', 'Vision change'], ['speech_change', 'Speech change'],
  ['cognitive_change', 'Cognitive change'], ['other', 'Other'],
];
const SEQUENCES = ['T1C', 'T1', 'T2', 'FLAIR'];
const DICOM_SEQUENCE_OPTIONS = ['T1', 'T1C', 'T2', 'FLAIR', 'OTHER', 'NOT_USABLE'];

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

function StepRail({ stage }) {
  const order = ['details','upload','qc','eligible','processing','complete'];
  const current = Math.max(0, order.indexOf(stage));
  const steps = [['01','Intake'],['02','MRI upload'],['03','QC & mapping'],['04','AI eligibility'],['05','Processing'],['06','Viewer']];
  return <div className="intake-step-rail">{steps.map((item,index)=><div key={item[0]} className={index < current ? 'intake-step intake-step--done' : index === current ? 'intake-step intake-step--active':'intake-step'}><span>{index < current ? <Icon name="check" size={13}/> : item[0]}</span><strong>{item[1]}</strong>{index < steps.length-1 ? <i/>:null}</div>)}</div>;
}

function StatusCard({ title, value, tone='neutral', detail }) {
  return <div className={`intake-status-card intake-status-card--${tone}`}><span>{title}</span><strong>{value}</strong>{detail ? <small>{detail}</small>:null}</div>;
}

export default function AnalysisIntakeWorkspace() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const reduceMotion = useReducedMotion();
  const pollRef = useRef(null);
  const [stage,setStage] = useState('details');
  const [busy,setBusy] = useState(false);
  const [error,setError] = useState('');
  const [notice,setNotice] = useState('');
  const [studyUuid,setStudyUuid] = useState(null);
  const [caseReference,setCaseReference] = useState(null);
  const [upload,setUpload] = useState(null);
  const [qc,setQc] = useState(null);
  const [capabilities,setCapabilities] = useState(null);
  const [series,setSeries] = useState([]);
  const [brainConfirmed,setBrainConfirmed] = useState(false);
  const [mapping,setMapping] = useState({ t1:'', t1c:'', t2:'', flair:'' });
  const [seriesMapping,setSeriesMapping] = useState({});
  const [job,setJob] = useState(null);
  const [processingLog,setProcessingLog] = useState([]);
  const [form,setForm] = useState({
    case_reference:'', patient_name:'', age_years:'', sex:'unknown',
    mri_date:new Date().toISOString().slice(0,10), symptoms:[], symptom_duration:'',
    prior_treatment:false, clinical_notes:'',
  });

  useEffect(()=>{
    if (searchParams.get('resume') !== '1') return;
    const active = getActiveCase();
    if (active?.studyUuid) {
      setStudyUuid(active.studyUuid); setCaseReference(active.caseReference);
      setStage(active.stage === 'viewer_ready' ? 'complete' : (active.stage || 'upload'));
      setNotice(`Resumed ${active.caseReference}. Internal study routing was restored automatically.`);
    }
  },[searchParams]);

  useEffect(()=>()=>{ if (pollRef.current) window.clearTimeout(pollRef.current); },[]);

  const volumes = useMemo(()=>qc?.checks?.volumes || [],[qc]);
  const threeDEligible = capabilities?.capabilities?.three_d_segmentation?.state === 'eligible';

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
      setNotice(result.patient_reused ? `Existing case reference ${result.case_reference} reused; a new MRI assessment was created.` : `Case ${result.case_reference} created. Technical study IDs are managed internally.`);
    } catch(reason){ setError(reason?.message || 'Could not create the analysis intake.'); }
    finally{ setBusy(false); }
  }

  async function uploadAndQc(){
    if (!upload || !studyUuid) return;
    setBusy(true); setError(''); setNotice('');
    try {
      const uploaded=await uploadStudySource(studyUuid,upload);
      const result=await runStudyQc(studyUuid);
      setQc(result); setStage('qc');
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
      if (routed.capabilities?.three_d_segmentation?.state !== 'eligible') setNotice('The study was routed safely, but 3D segmentation is not currently eligible. Review the reasons below.');
      else setNotice('Required 3D MRI inputs passed capability routing. The study is ready for guarded segmentation preparation.');
    } catch(reason){ setError(reason?.message || 'Sequence/scope confirmation or capability routing failed.'); }
    finally{ setBusy(false); }
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

  function openViewer(){ patchActiveCase({stage:'viewer_ready'}); router.push('/viewer/current'); }

  return (
    <MotionConfig reducedMotion="user">
      <main className="intake-shell">
        <div className="ambient ambient--one"/><div className="ambient ambient--two"/>
        <header className="intake-topbar"><Link href="/" className="intake-back"><Icon name="back"/>Home</Link><div className="intake-brand"><span className="intake-brand-mark"><Icon name="brain"/></span><div><span className="eyebrow">NEUROGLIOMA AI</span><strong>New MRI Analysis</strong></div></div><div className="intake-security"><Icon name="shield"/><span>MRI analysis workflow</span></div></header>
        <StepRail stage={stage}/>
        <section className="intake-stage-wrap">
          <AnimatePresence mode="wait">
            {stage==='details' ? <motion.div key="details" className="intake-stage-card glass-panel" initial={{opacity:0,y:10}} animate={{opacity:1,y:0}} exit={{opacity:0,y:-8}}>
              <div className="intake-stage-heading"><div><span className="eyebrow">STEP 01 · CLINICAL CONTEXT</span><h1>Start a new MRI analysis</h1><p>Enter the clinical context for this MRI assessment.</p></div><span className="intake-stage-icon"><Icon name="brain" size={26}/></span></div>
              <form onSubmit={createCase} className="intake-form">
                <div className="field-grid field-grid--3"><label><span>Case reference <small>optional</small></span><input value={form.case_reference} onChange={(e)=>updateForm('case_reference',e.target.value)} placeholder="Auto-generate if blank"/></label><label><span>Patient name <small>optional</small></span><input value={form.patient_name} onChange={(e)=>updateForm('patient_name',e.target.value)} placeholder="De-identified / local use"/></label><label><span>MRI date</span><input type="date" required value={form.mri_date} onChange={(e)=>updateForm('mri_date',e.target.value)}/></label></div>
                <div className="field-grid field-grid--3"><label><span>Age</span><input type="number" min="18" max="100" value={form.age_years} onChange={(e)=>updateForm('age_years',e.target.value)} placeholder="Optional"/></label><label><span>Sex</span><select value={form.sex} onChange={(e)=>updateForm('sex',e.target.value)}><option value="unknown">Unknown</option><option value="female">Female</option><option value="male">Male</option><option value="other">Other</option></select></label><label className="toggle-field"><span>Prior brain tumor treatment?</span><button type="button" className={form.prior_treatment?'yes':'no'} onClick={()=>updateForm('prior_treatment',!form.prior_treatment)}><i/>{form.prior_treatment?'Yes — outside V1 scope':'No'}</button></label></div>
                <div className="field-block"><span>Presenting symptoms</span><div className="symptom-chips">{SYMPTOMS.map(([value,label])=><button type="button" key={value} className={form.symptoms.includes(value)?'active':''} onClick={()=>toggleSymptom(value)}>{label}</button>)}</div></div>
                <div className="field-grid field-grid--2"><label><span>Symptom duration</span><input value={form.symptom_duration} onChange={(e)=>updateForm('symptom_duration',e.target.value)} placeholder="e.g. 3 weeks"/></label><label><span>Clinical note <small>optional</small></span><input value={form.clinical_notes} onChange={(e)=>updateForm('clinical_notes',e.target.value)} placeholder="Brief context only"/></label></div>
                <div className="intake-form-foot"><div/><motion.button className="intake-primary" type="submit" disabled={busy} whileTap={reduceMotion?undefined:{scale:.98}}>{busy?'Creating analysis…':'Create analysis'}<Icon name="arrow"/></motion.button></div>
              </form>
            </motion.div> : null}

            {stage==='upload' ? <motion.div key="upload" className="intake-stage-card glass-panel" initial={{opacity:0,y:10}} animate={{opacity:1,y:0}}>
              <div className="intake-stage-heading"><div><span className="eyebrow">STEP 02 · MRI INTAKE</span><h1>Upload brain MRI</h1><p>Case <strong>{caseReference}</strong> is ready for MRI upload.</p></div><span className="intake-stage-icon"><Icon name="upload" size={26}/></span></div>
              <label className={upload?'upload-dropzone upload-dropzone--ready':'upload-dropzone'}><input type="file" accept=".zip,.nii,.gz,.dcm,.dicom,.ima,.jpg,.jpeg,.png" onChange={(e)=>setUpload(e.target.files?.[0] || null)}/><span className="upload-orbit"><Icon name="upload" size={27}/></span><strong>{upload?upload.name:'Drop or choose one MRI package'}</strong><p>{upload?`${(upload.size/1024/1024).toFixed(1)} MB selected`:'DICOM ZIP, multi-volume NIfTI ZIP, or supported standalone MRI image'}</p><small>Format detection is automatic. Do not mix DICOM and NIfTI in one archive.</small></label>
              <div className="intake-action-row"><Link href="/" className="intake-secondary">Cancel</Link><button className="intake-primary" disabled={!upload || busy} onClick={uploadAndQc}>{busy?'Uploading + validating…':'Upload & run MRI QC'}<Icon name="arrow"/></button></div>
            </motion.div> : null}

            {stage==='qc' ? <motion.div key="qc" className="intake-stage-card intake-stage-card--wide glass-panel" initial={{opacity:0,y:10}} animate={{opacity:1,y:0}}>
              <div className="intake-stage-heading"><div><span className="eyebrow">STEP 03 · QUALITY & SEQUENCE REVIEW</span><h1>Confirm imaging context</h1><p>AI processing remains blocked until brain scope and required sequence mapping are explicit.</p></div><span className="intake-stage-icon"><Icon name="scan" size={26}/></span></div>
              <div className="intake-status-grid"><StatusCard title="QC status" value={String(qc?.qc_status || 'pending').toUpperCase()} tone={qc?.qc_status==='fail'?'bad':qc?.qc_status==='pass'?'good':'warn'}/><StatusCard title="Volumes" value={String(qc?.checks?.volume_count ?? series.length ?? 0)} detail={series.length?'DICOM series':'NIfTI volumes'}/><StatusCard title="Manual review" value={qc?.manual_review_required?'REQUIRED':'NOT REQUIRED'} tone={qc?.manual_review_required?'warn':'good'}/></div>
              {(qc?.fail_reasons?.length || qc?.warnings?.length || qc?.partial_reasons?.length) ? <div className="qc-messages">{qc?.fail_reasons?.map((x)=><span className="bad" key={x}><Icon name="alert" size={13}/>{x}</span>)}{qc?.partial_reasons?.map((x)=><span className="warn" key={x}><Icon name="alert" size={13}/>{x}</span>)}{qc?.warnings?.map((x)=><span key={x}>{x}</span>)}</div>:null}
              <label className={brainConfirmed?'brain-confirm brain-confirm--active':'brain-confirm'}><input type="checkbox" checked={brainConfirmed} onChange={(e)=>setBrainConfirmed(e.target.checked)}/><span><Icon name="brain"/></span><div><strong>I confirm this upload is a brain MRI study</strong></div><i><Icon name="check" size={13}/></i></label>
              {volumes.length ? <div className="mapping-panel"><div className="mapping-heading"><div><span className="eyebrow">NIFTI CHANNEL MAPPING</span><h2>Map the four MRI sequences</h2></div><p>Confirm the sequence mapping before analysis.</p></div><div className="mapping-grid">{[['t1c','T1C'],['t1','T1'],['t2','T2'],['flair','FLAIR']].map(([key,label])=><label key={key}><span>{label}</span><select value={mapping[key]} onChange={(e)=>setMapping((m)=>({...m,[key]:e.target.value}))}><option value="">Choose volume</option>{volumes.map((volume)=><option value={volume.volume_index} key={volume.volume_index}>Volume {Number(volume.volume_index)+1} · {(volume.shape||[]).join('×')} · {(volume.zooms||[]).slice(0,3).join('/')} mm</option>)}</select></label>)}</div></div>:null}
              {series.length ? <div className="mapping-panel"><div className="mapping-heading"><div><span className="eyebrow">DICOM SERIES REVIEW</span><h2>Confirm detected sequences</h2></div><p>Confirm the detected MRI sequence labels.</p></div><div className="dicom-series-list">{series.map((item)=><div key={item.id}><div><strong>Series {item.series_number ?? '—'}</strong><span>{item.slice_count} slices</span><small>{item.sequence_metadata?.series_description || item.sequence_metadata?.protocol_name || 'No description'}</small></div><select value={seriesMapping[item.id] || 'OTHER'} onChange={(e)=>setSeriesMapping((m)=>({...m,[item.id]:e.target.value}))}>{DICOM_SEQUENCE_OPTIONS.map((label)=><option key={label} value={label}>{label}</option>)}</select></div>)}</div></div>:null}
              <div className="intake-action-row"><button className="intake-secondary" onClick={()=>setStage('upload')}>Back</button><button className="intake-primary" disabled={busy || qc?.qc_status==='fail'} onClick={confirmAndRoute}>{busy?'Applying safety gates…':'Confirm & check AI eligibility'}<Icon name="arrow"/></button></div>
            </motion.div> : null}

            {stage==='eligible' ? <motion.div key="eligible" className="intake-stage-card glass-panel" initial={{opacity:0,y:10}} animate={{opacity:1,y:0}}>
              <div className="intake-stage-heading"><div><span className="eyebrow">STEP 04 · CAPABILITY ROUTING</span><h1>{threeDEligible?'3D analysis is eligible':'Manual review required'}</h1><p>Available analysis options are based on the validated MRI input.</p></div><span className={`intake-stage-icon ${threeDEligible?'good':'warn'}`}><Icon name={threeDEligible?'check':'alert'} size={26}/></span></div>
              <div className="capability-list">{Object.entries(capabilities?.capabilities || {}).map(([key,value])=><div key={key}><span className={`capability-state capability-state--${value.state}`}>{value.state}</span><div><strong>{key.replaceAll('_',' ')}</strong><small>{value.user_message || value.reasons?.[0] || 'No additional note'}</small></div></div>)}</div>
              <div className="intake-action-row"><button className="intake-secondary" onClick={()=>setStage('qc')}>Review mapping</button><button className="intake-primary" disabled={!threeDEligible || busy} onClick={startProcessing}>Prepare & queue 3D analysis<Icon name="arrow"/></button></div>
            </motion.div> : null}

            {stage==='processing' ? <motion.div key="processing" className="intake-stage-card glass-panel" initial={{opacity:0,y:10}} animate={{opacity:1,y:0}}>
              <div className="intake-stage-heading"><div><span className="eyebrow">STEP 05 · GUARDED AI PROCESSING</span><h1>Analyzing multimodal MRI</h1><p>The MRI study is being prepared and analyzed.</p></div><span className="intake-stage-icon processing"><Icon name="scan" size={26}/></span></div>
              <div className="processing-visual"><div className="processing-radar"><span/><i/><b/></div><div><strong>{job?.status==='running'?'MRI analysis is running':job?.status==='queued'?'Analysis queued':'Preparing MRI data'}</strong><small>{job ? `Processing attempt ${job.attempts}/${job.max_attempts}` : 'Preparing analysis'}</small></div></div>
              <div className="processing-log">{processingLog.map((item,index)=><div key={`${item.at}-${index}`} className={`processing-log-row processing-log-row--${item.status}`}><span>{item.status==='done'?<Icon name="check" size={13}/>:item.status==='warning'?<Icon name="alert" size={13}/>:<i/>}</span><strong>{item.label}</strong></div>)}</div>
              
            </motion.div> : null}

            {stage==='complete' ? <motion.div key="complete" className="intake-stage-card intake-complete glass-panel" initial={{opacity:0,scale:.985}} animate={{opacity:1,scale:1}}>
              <div className="intake-success-orbit"><span><Icon name="check" size={30}/></span></div><span className="eyebrow">ANALYSIS READY</span><h1>{caseReference || getActiveCase()?.caseReference}</h1><p>The MRI analysis is ready for review.</p><div className="complete-badges"><span><Icon name="scan" size={15}/>3-plane MRI</span><span>WT / TC / ET</span><span><Icon name="shield" size={15}/>Checksummed assets</span></div><button className="intake-primary intake-primary--large" onClick={openViewer}>Open clinical viewer <Icon name="arrow"/></button>
            </motion.div> : null}
          </AnimatePresence>
          {notice ? <div className="intake-banner intake-banner--info">{notice}</div>:null}
          {error ? <div className="intake-banner intake-banner--error"><Icon name="alert" size={15}/><span>{error}</span></div>:null}
        </section>
        <footer className="intake-footer"><span>NeuroGlioma AI</span></footer>
      </main>
    </MotionConfig>
  );
}
