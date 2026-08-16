'use client';

import Link from 'next/link';
import { useEffect, useState } from 'react';
import { motion, MotionConfig, useReducedMotion } from 'motion/react';
import { getRecentCases, setActiveCase } from '@/lib/activeCase';

function BrandMark() {
  return <div className="brand-mark" aria-hidden="true"><span className="brand-mark__ring"/><span className="brand-mark__cross brand-mark__cross--v"/><span className="brand-mark__cross brand-mark__cross--h"/><span className="brand-mark__core"/></div>;
}

function FlowNode({ index, label, detail, active }) {
  return <div className={active ? 'home-flow-node home-flow-node--active' : 'home-flow-node'}><span>{index}</span><div><strong>{label}</strong><small>{detail}</small></div></div>;
}

export default function HomePage() {
  const reduceMotion = useReducedMotion();
  const [recent, setRecent] = useState([]);
  useEffect(() => setRecent(getRecentCases()), []);

  function resumeCase(item) {
    setActiveCase(item);
  }

  return (
    <MotionConfig reducedMotion="user">
      <main className="ng-home-shell">
        <div className="ambient ambient--one"/><div className="ambient ambient--two"/><div className="ng-grid-glow"/>
        <header className="ng-home-nav">
          <div className="ng-home-brand"><BrandMark/><div><span className="eyebrow">AI-ASSISTED NEURO-ONCOLOGY WORKSPACE</span><strong>NeuroGlioma AI</strong></div></div>
          <div className="ng-nav-actions"><Link href="/analysis/new" className="ng-nav-button">New analysis</Link></div>
        </header>

        <section className="ng-hero">
          <motion.div className="ng-hero-copy" initial={reduceMotion ? {opacity:0}:{opacity:0,y:18}} animate={{opacity:1,y:0}} transition={{duration:.55}}>
            <span className="ng-kicker"><i/> MULTIMODAL MRI · SEGMENTATION · QUANTIFICATION</span>
            <h1>Clinical MRI intelligence,<br/><em>built for review.</em></h1>
            <p>Upload a compatible brain MRI study, validate its imaging context, run guarded AI analysis, and review tumor segmentation, measurements and atlas-derived localization in one workflow.</p>
            <div className="ng-hero-actions"><Link href="/analysis/new" className="ng-primary-cta">Start new MRI analysis <span>→</span></Link><a href="#workflow" className="ng-secondary-cta">See workflow</a></div>
            <div className="ng-trust-row"><span>4-channel MRI</span><span>WT / TC / ET</span><span>3-plane MPR</span><span>Human review</span></div>
          </motion.div>

          <motion.div className="ng-hero-console glass-panel" initial={reduceMotion ? {opacity:0}:{opacity:0,scale:.97,x:18}} animate={{opacity:1,scale:1,x:0}} transition={{duration:.65,delay:.08}}>
            <div className="ng-console-top"><div><span className="eyebrow">ANALYSIS PIPELINE</span><strong>MRI analysis pipeline</strong></div><span className="ng-console-status"><i/> Ready</span></div>
            <div className="ng-brain-orbit" aria-hidden="true"><div className="ng-orbit ng-orbit--1"/><div className="ng-orbit ng-orbit--2"/><div className="ng-orbit ng-orbit--3"/><div className="ng-brain-core"><span/><span/><span/><span/></div><div className="ng-scan-line"/></div>
            <div className="ng-flow-stack"><FlowNode index="01" label="Intake & MRI QC" detail="Format, scope, sequence and spatial checks" active/><FlowNode index="02" label="3D AI analysis" detail="SegResNet · WT / TC / ET"/><FlowNode index="03" label="Clinical review" detail="MPR overlays · volume · atlas context"/></div>
            
          </motion.div>
        </section>

        <section id="workflow" className="ng-feature-strip">
          <article><span>01</span><div><strong>One guided intake</strong><p>Clinical context, MRI upload and guided study creation.</p></div></article>
          <article><span>02</span><div><strong>Safety-gated processing</strong><p>QC and capability routing decide what the current input can safely support.</p></div></article>
          <article><span>03</span><div><strong>Clinician-first viewer</strong><p>Orthographic MRI review, segmentation editing, quantitative measurements and provenance.</p></div></article>
        </section>

        {recent.length ? <section className="ng-recent-section"><div className="ng-section-title"><div><h2>Recent analyses</h2></div></div><div className="ng-recent-grid">{recent.map((item) => <Link key={item.studyUuid} className="ng-recent-card" href={item.stage === 'viewer_ready' ? '/viewer/current' : '/analysis/new?resume=1'} onClick={() => resumeCase(item)} aria-label={`Continue ${item.caseReference}`}><div><span className="status-dot"/><strong>{item.caseReference}</strong><small>{item.sourceFormat ? item.sourceFormat.toUpperCase() : 'MRI intake'} · {item.stage || 'created'}</small></div><span className="ng-recent-card__action">Continue <span>→</span></span></Link>)}</div></section> : null}

        <footer className="ng-home-footer"><span>NeuroGlioma AI</span></footer>
      </main>
    </MotionConfig>
  );
}
